import os
import importlib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    drive = importlib.import_module("google.colab").drive
except ModuleNotFoundError:
    drive = None

if drive is not None:
    drive.mount('/content/drive', force_remount=True)

DATA_DIR = '/content/drive/MyDrive/TimeSeriesDeepLearning_FIM601/kaggle_data/optiver-realized-volatility-prediction'
DIR = '/content/data'
rng = np.random.default_rng()

class OrderFlowDataset(Dataset):
    def __init__(self, target_csv_path: str, book_path: str, trade_path: str, transform=None):
        self.target = pd.read_csv(target_csv_path)
        self.index_map = self.target[['stock_id', 'time_id']].to_numpy()
        self.target = self.target.to_numpy()

        self.transform = transform

        full_book = pd.read_parquet(book_path)
        full_trade = pd.read_parquet(trade_path)

        self.books = {k : v for k, v in full_book.groupby(['stock_id', 'time_id'])}
        self.trades = {k : v for k, v in full_trade.groupby(['stock_id', 'time_id'])}

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        stock_id, time_id = self.index_map[idx]
        book_data = self.books.get((stock_id, time_id), pd.DataFrame())
        trade_data = self.trades.get((stock_id, time_id), pd.DataFrame())
        target = self.target[idx, 2]

        sample = {"book": book_data, "trade": trade_data, "r_vol": target}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToImage(object):
    def __init__(self, output_size=(600, 600, 4), bounds="auto", pad = True, include_features=True):
        self.output_size = output_size
        self.bounds = bounds
        self.pad = pad
        self.include_features = include_features

    @staticmethod
    def _safe_std(values: np.ndarray) -> float:
        if values.size < 2:
            return 0.0
        return float(np.std(values, ddof=1))

    def _extract_engineered_features(self, book: pd.DataFrame, trade: pd.DataFrame) -> np.ndarray:
        if book.empty:
            return np.zeros(16, dtype=np.float32)

        bid_price1 = book["bid_price1"].to_numpy(dtype=np.float64)
        ask_price1 = book["ask_price1"].to_numpy(dtype=np.float64)
        bid_size1 = book["bid_size1"].to_numpy(dtype=np.float64)
        ask_size1 = book["ask_size1"].to_numpy(dtype=np.float64)
        bid_size2 = book["bid_size2"].to_numpy(dtype=np.float64)
        ask_size2 = book["ask_size2"].to_numpy(dtype=np.float64)
        seconds = book["seconds_in_bucket"].to_numpy(dtype=np.int32)

        denom = bid_size1 + ask_size1 + 1e-8
        wap1 = (bid_price1 * ask_size1 + ask_price1 * bid_size1) / denom
        spread1 = ask_price1 - bid_price1

        # Realized volatility from tick-by-tick log-return sequence.
        wap1_clipped = np.clip(wap1, 1e-8, None)
        if wap1_clipped.size > 1:
            log_ret = np.diff(np.log(wap1_clipped))
            realized_vol = float(np.sqrt(np.sum(log_ret * log_ret)))
        else:
            realized_vol = 0.0

        vol_l1 = bid_size1 + ask_size1
        vol_l2 = bid_size2 + ask_size2
        vol_total = vol_l1 + vol_l2
        vol_imbalance = (bid_size1 - ask_size1) / (bid_size1 + ask_size1 + 1e-8)

        book_update_count = float(len(book))
        unique_seconds_count = float(np.unique(seconds).size)
        mean_spread = float(np.mean(spread1))
        std_spread = self._safe_std(spread1)
        wap1_mean = float(np.mean(wap1))
        vol_sum = float(np.sum(vol_total))
        vol_q90 = float(np.quantile(vol_total, 0.9)) if vol_total.size else 0.0
        vol_imbalance_mean = float(np.mean(vol_imbalance))
        vol_imbalance_std = self._safe_std(vol_imbalance)

        trade_count = 0.0
        trade_size_sum = 0.0
        trade_order_count_sum = 0.0
        trade_avg_size_per_order = 0.0
        trade_inter_arrival_mean = 0.0
        trade_price_return_std = 0.0

        if not trade.empty:
            trade_count = float(len(trade))
            trade_seconds = trade["seconds_in_bucket"].to_numpy(dtype=np.int32)
            trade_size = trade["size"].to_numpy(dtype=np.float64)
            trade_order_count = trade["order_count"].to_numpy(dtype=np.float64)
            trade_price = trade["price"].to_numpy(dtype=np.float64)

            trade_size_sum = float(np.sum(trade_size))
            trade_order_count_sum = float(np.sum(trade_order_count))
            trade_avg_size_per_order = float(np.mean(trade_size / (trade_order_count + 1e-8)))

            if trade_seconds.size > 1:
                trade_inter_arrival_mean = float(np.mean(np.diff(trade_seconds)))

            if trade_price.size > 1:
                trade_price_ret = np.diff(np.log(np.clip(trade_price, 1e-8, None)))
                trade_price_return_std = self._safe_std(trade_price_ret)

        features = np.array([
            book_update_count,
            unique_seconds_count,
            mean_spread,
            std_spread,
            wap1_mean,
            realized_vol,
            vol_sum,
            vol_q90,
            vol_imbalance_mean,
            vol_imbalance_std,
            trade_count,
            trade_size_sum,
            trade_order_count_sum,
            trade_avg_size_per_order,
            trade_inter_arrival_mean,
            trade_price_return_std,
        ], dtype=np.float32)
        return features

    def __call__(self, sample):
        book, trade, r_vol = sample["book"], sample["trade"], sample["r_vol"]
        n_time, n_price, n_channels = self.output_size
        image = np.zeros((n_time, n_price, n_channels), dtype=np.int32)
        features = self._extract_engineered_features(book, trade) if self.include_features else None
        if book.empty:
            output = {"image": image, "r_vol": r_vol}
            if features is not None:
                output["features"] = features
            return output

        # Compute per-sample bounds when set to auto.
        if self.bounds == "auto":
            min_price = min(book["bid_price1"].min(), book["bid_price2"].min(),
                            book["ask_price1"].min(), book["ask_price2"].min())
            max_price = max(book["bid_price1"].max(), book["bid_price2"].max(),
                            book["ask_price1"].max(), book["ask_price2"].max())

            if not trade.empty:
                min_price = min(min_price, trade["price"].min())
                max_price = max(max_price, trade["price"].max())

            padding = 0.001 * (max_price - min_price + 1e-8)
            bounds = (float(min_price - padding), float(max_price + padding))
        else:
            bounds = (float(self.bounds[0]), float(self.bounds[1]))

        price_edges = np.linspace(bounds[0], bounds[1], n_price + 1)

        # Channels 0 and 1 (book): place size at exact observed second and price bin.
        bid_bin_1 = np.searchsorted(price_edges, book["bid_price1"].to_numpy()) - 1
        bid_bin_2 = np.searchsorted(price_edges, book["bid_price2"].to_numpy()) - 1
        ask_bin_1 = np.searchsorted(price_edges, book["ask_price1"].to_numpy()) - 1
        ask_bin_2 = np.searchsorted(price_edges, book["ask_price2"].to_numpy()) - 1

        bid_bin_1 = np.clip(bid_bin_1, 0, n_price - 1)
        bid_bin_2 = np.clip(bid_bin_2, 0, n_price - 1)
        ask_bin_1 = np.clip(ask_bin_1, 0, n_price - 1)
        ask_bin_2 = np.clip(ask_bin_2, 0, n_price - 1)

        sec = book["seconds_in_bucket"].to_numpy(dtype=np.int32)

        bs1 = book["bid_size1"].to_numpy(dtype=np.int32)
        bs2 = book["bid_size2"].to_numpy(dtype=np.int32)
        as1 = book["ask_size1"].to_numpy(dtype=np.int32)
        as2 = book["ask_size2"].to_numpy(dtype=np.int32)

        np.add.at(image, (sec, bid_bin_1, 0), bs1)  # type: ignore[arg-type]
        np.add.at(image, (sec, bid_bin_2, 0), bs2)  # type: ignore[arg-type]
        np.add.at(image, (sec, ask_bin_1, 1), as1)  # type: ignore[arg-type]
        np.add.at(image, (sec, ask_bin_2, 1), as2)  # type: ignore[arg-type]

        # Channels 2 and 3 (trade): place size at exact observed second and price bin.
        if not trade.empty:
            trade_sec = trade["seconds_in_bucket"].to_numpy(dtype=np.int32)
            trade_bin = np.searchsorted(price_edges, trade["price"].to_numpy(), side="right") - 1
            trade_bin = np.clip(trade_bin, 0, n_price - 1)
            trade_size = trade["size"].to_numpy(dtype=np.int32)
            trade_oc = trade["order_count"].to_numpy(dtype=np.int32)

            np.add.at(image[:, :, 2], (trade_sec, trade_bin), trade_size)
            if self.pad:
                left_mask = trade_bin - 1 >= 0
                right_mask = trade_bin + 1 < n_price
                np.add.at(image[:, :, 2], (trade_sec[left_mask], trade_bin[left_mask] - 1), trade_size[left_mask])
                np.add.at(image[:, :, 2], (trade_sec[right_mask], trade_bin[right_mask] + 1), trade_size[right_mask])
            np.add.at(image[:, :, 3], (trade_sec, trade_bin), trade_oc)
        

        output = {"image": image, "r_vol": r_vol}
        if features is not None:
            output["features"] = features
        return output