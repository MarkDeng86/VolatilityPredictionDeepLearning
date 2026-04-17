# %%
import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import timm

from ImageModelUtils import OrderFlowRegressor, RelativeHuberLoss
from ImageEncoding import OrderFlowDataset, ToImage

# from google.colab import drive
# drive.mount('/content/drive')

DATA_DIR = '/content/drive/MyDrive/TimeSeriesDeepLearning_FIM601/kaggle_data/optiver-realized-volatility-prediction'
DIR = '/content/drive/MyDrive/TimeSeriesDeepLearning_FIM601/'
LOCAL_DATA_DIR = '/content/data'

# Standard Python check and copy
if not os.path.exists(LOCAL_DATA_DIR):
    print(f"Copying data from {DATA_DIR} to {LOCAL_DATA_DIR}...")
    shutil.copytree(DATA_DIR, LOCAL_DATA_DIR)
    print("Copy complete.")
else:
    print("Local data directory already exists. Skipping copy.")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = ToImage(output_size=(600, 160, 3))
train_dataset = OrderFlowDataset(
    f"{LOCAL_DATA_DIR}/train.csv", 
    f"{LOCAL_DATA_DIR}/book_train.parquet", 
    f"{LOCAL_DATA_DIR}/trade_train.parquet", 
    transform=img_transform
)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# %%
hyperparameters = {
    "batch_size": 256, 
    "num_workers": os.cpu_count() - 1 if os.cpu_count() else 0, 
    "pin_memory": True, 
    "prefetch_factor": 2
}

# %%
def main():
    train_loader = DataLoader(train_dataset, shuffle=True, **hyperparameters)
    val_loader = DataLoader(val_dataset, shuffle=False, **hyperparameters)
    model = OrderFlowRegressor().to(device)
    criterion = RelativeHuberLoss()

    muon_params = [p for p in model.parameters() if p.ndim == 2]
    adamw_params = [p for p in model.parameters() if p.ndim != 2]

    optimizer_adamw = optim.AdamW(adamw_params, lr=15e-6, weight_decay=0.05)
    optimizer_muon = optim.Muon(muon_params, lr=0.0005, momentum=0.95)

    num_epochs = 20
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    decay_steps = total_steps - warmup_steps

    def build_scheduler(optimizer):
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    scheduler_adamw = build_scheduler(optimizer_adamw)
    scheduler_muon = build_scheduler(optimizer_muon)

    checkpoint_path = f"{DIR}/checkpoint.pth"
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    test_losses = []

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_adamw.load_state_dict(checkpoint['optimizer_adamw_state_dict'])
        optimizer_muon.load_state_dict(checkpoint['optimizer_muon_state_dict'])
        scheduler_adamw.load_state_dict(checkpoint['scheduler_adamw_state_dict'])
        scheduler_muon.load_state_dict(checkpoint['scheduler_muon_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['val_losses']
        print(f"Resuming from epoch {start_epoch + 1}...")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            
            for batch in train_loader:
                # Extract tensors from the dictionary
                images = batch["image"]
                targets = batch["r_vol"]
                
                # 1. Rearrange from (B, H, W, C) to (B, C, H, W) -> [512, 4, 600, 80]
                images = images.permute(0, 3, 1, 2)
                    
                # 2. Cast from np.int32 to float32 and move to GPU
                images = images.to(device, dtype=torch.float32, non_blocking=True)
                targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
                
                optimizer_adamw.zero_grad()
                optimizer_muon.zero_grad()
                
                with autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(images)
                
                loss = criterion(outputs.float(), targets)
                loss.backward()    
                
                optimizer_adamw.step()
                optimizer_muon.step()
                
                scheduler_adamw.step()
                scheduler_muon.step()
                
                running_loss += loss.item() * images.size(0)
                
            epoch_loss = running_loss / len(train_dataset)
            train_losses.append(epoch_loss)
            print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {epoch_loss:.6f}")

            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader: 
                    images = batch["image"]
                    targets = batch["r_vol"]
                    
                    images = images.permute(0, 3, 1, 2)
                        
                    images = images.to(device, dtype=torch.float32, non_blocking=True)
                    targets = torch.as_tensor(targets, dtype=torch.float32, device = device)
                    
                    with autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(images)
                        val_loss = criterion(outputs, targets)
                        
                    running_val_loss += val_loss.item() * images.size(0)

            val_loss = running_val_loss / len(val_dataset)
            test_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{num_epochs} | Validation Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{DIR}/best_model.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_adamw_state_dict': optimizer_adamw.state_dict(),
                'optimizer_muon_state_dict': optimizer_muon.state_dict(),
                'scheduler_adamw_state_dict': scheduler_adamw.state_dict(),
                'scheduler_muon_state_dict': scheduler_muon.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': test_losses
            }, checkpoint_path)
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
    finally:
        if train_losses:
            min_len = min(len(train_losses), len(test_losses))
            history_df = pd.DataFrame({
                'epoch': list(range(1, min_len + 1)),
                'train_loss': train_losses[:min_len],
                'val_loss': test_losses[:min_len]
            })
            history_df.to_csv(f"{DIR}/training_history.csv", index=False)

# %%
if __name__ == "__main__":
    main()


