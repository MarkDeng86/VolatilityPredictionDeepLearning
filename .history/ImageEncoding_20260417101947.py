import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


rng = np.random.default_rng()

class OrderFlowDataset(Dataset):
    def __init__(self, target_csv_path, book_path: str, trade_path: str, transform=None):
        full_book = pd.read_parquet(book_path)
        full_trade = pd.read_parquet(trade_path)

        self.books = {k : v for k, v in full_book.groupby(['stock_id', 'time_id'])}
        self.trades = {k : v for k, v in full_trade.groupby(['stock_id', 'time_id'])}

        if target_csv_path is not None:
            self.target = pd.read_csv(target_csv_path)
            self.index_map = self.target[['stock_id', 'time_id']].to_numpy()
            self.target = self.target.to_numpy()
        else: 
            self.target = None
            self.index_map = list(self.books.keys())

        self.transform = transform


    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        stock_id, time_id = self.index_map[idx]
        book_data = self.books.get((stock_id, time_id), pd.DataFrame())
        trade_data = self.trades.get((stock_id, time_id), pd.DataFrame())
        target = self.target[idx, 2] if self.target is not None else None

        sample = {"book": book_data, "trade": trade_data, "r_vol": target}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToImage(object):
    def __init__(self, output_size=(600, 160, 3), bounds="auto", pad = True):
        self.output_size = output_size
        self.bounds = bounds
        self.pad = pad

    def __call__(self, sample):
        book, trade, r_vol = sample["book"], sample["trade"], sample["r_vol"]
        n_time, n_price, n_channels = self.output_size
        image = np.zeros((n_time, n_price, n_channels), dtype=np.float32)
        if book.empty:
            return {"image": image, "r_vol": r_vol}

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
            bounds = (min_price - padding, max_price + padding)
        else:
            bounds = self.bounds

        price_edges = np.linspace(bounds[0], bounds[1], n_price + 1)

        # Channels 0 and 1 (book): place size at exact observed second and price bin.
        bid_bin_1 = np.searchsorted(price_edges, book["bid_price1"].to_numpy()) - 1
        bid_bin_2 = np.searchsorted(price_edges, book["bid_price2"].to_numpy()) - 1
        ask_bin_1 = np.searchsorted(price_edges, book["ask_price1"].to_numpy()) - 1
        ask_bin_2 = np.searchsorted(price_edges, book["ask_price2"].to_numpy()) - 1

        sec = book["seconds_in_bucket"].to_numpy(dtype=np.int32)

        bs1 = book["bid_size1"].to_numpy(dtype=np.float32)
        bs2 = book["bid_size2"].to_numpy(dtype=np.float32)
        as1 = book["ask_size1"].to_numpy(dtype=np.float32)
        as2 = book["ask_size2"].to_numpy(dtype=np.float32)

        np.add.at(image, (sec, bid_bin_1, 0), bs1)
        np.add.at(image, (sec, bid_bin_2, 0), bs2)
        np.add.at(image, (sec, ask_bin_1, 1), as1)
        np.add.at(image, (sec, ask_bin_2, 1), as2)

        # Channels 2 and 3 (trade): place size at exact observed second and price bin.
        if not trade.empty:
            trade_sec = trade["seconds_in_bucket"].to_numpy(dtype=np.int32)
            trade_bin = np.searchsorted(price_edges, trade["price"].to_numpy(), side="right") - 1
            trade_size = trade["size"].to_numpy(dtype=np.float32)
            trade_oc = trade["order_count"].to_numpy(dtype=np.float32)
            oc_grid = np.zeros((n_time, n_price), dtype=np.float32)

            np.add.at(image[:, :, 2], (trade_sec, trade_bin), trade_size)
            np.add.at(oc_grid, (trade_sec, trade_bin), trade_oc)
            if self.pad:
                left_mask = trade_bin - 1 >= 0
                right_mask = trade_bin + 1 < n_price
                np.add.at(image[:, :, 2], (trade_sec[left_mask], trade_bin[left_mask] - 1), trade_size[left_mask])
                np.add.at(image[:, :, 2], (trade_sec[right_mask], trade_bin[right_mask] + 1), trade_size[right_mask])

            np.divide(image[:, :, 2], oc_grid, out=image[:, :, 2], where=oc_grid > 0)
        

        return {"image": image, "r_vol": r_vol}