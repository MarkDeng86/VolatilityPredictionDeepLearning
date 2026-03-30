import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import timm

from ImageEncoding import OrderFlowDataset, ToImage

DATA_DIR = '/content/drive/MyDrive/TimeSeriesDeepLearning_FIM601/kaggle_data/optiver-realized-volatility-prediction'

class OrderFlowRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=True, 
            in_chans=4, 
            num_classes=1, 
            dynamic_img_size=True
        )

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 8)) 
        return self.model(x).squeeze(-1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_transform = ToImage(output_size=(600, 80, 4))
    train_dataset = OrderFlowDataset(
        f"{DATA_DIR}/train.csv", 
        f"{DATA_DIR}/book_train.parquet", 
        f"{DATA_DIR}/trade_train.parquet", 
        transform=img_transform
    )
    
    hyperparameters = {
        "batch_size": 512, 
        "num_workers": os.cpu_count() - 1 if os.cpu_count() else 0, 
        "pin_memory": True, 
        "prefetch_factor": 2
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **hyperparameters)

    model = OrderFlowRegressor().to(device)
    criterion = nn.HuberLoss()

    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    adamw_params = [p for p in model.parameters() if p.ndim < 2]
    
    optimizer_adamw = optim.AdamW(adamw_params, lr=1e-4, weight_decay=1e-4)
    optimizer_muon = optim.Muon(muon_params, lr=0.02, momentum=0.95)

    num_epochs = 10
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    decay_steps = total_steps - warmup_steps

    def build_scheduler(optimizer):
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    scheduler_adamw = build_scheduler(optimizer_adamw)
    scheduler_muon = build_scheduler(optimizer_muon)

    scaler = GradScaler('cuda')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            if images.shape[-1] == 4: 
                images = images.permute(0, 3, 1, 2)
                
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, dtype=torch.float32, non_blocking=True)
            
            optimizer_adamw.zero_grad()
            optimizer_muon.zero_grad()
            
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            
            scaler.step(optimizer_adamw)
            scaler.step(optimizer_muon)
            scaler.update()
            
            scheduler_adamw.step()
            scheduler_muon.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {epoch_loss:.6f}")

if __name__ == "__main__":
    main()