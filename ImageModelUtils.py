import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class OrderFlowRegressor(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()
        self.model = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=True, 
            in_chans=3, 
            num_classes=0, 
            global_pool='avg',
            dynamic_img_size=True
        )
        image_dim = self.model.num_features

        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(tabular_dim),
            nn.Linear(tabular_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(image_dim + 128, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x, features):
        # x is expected to be (B, C, H, W)
        # Log-transform the input
        x = torch.log1p(x)
        mean = x.mean(dim = [-2, -1], keepdim=True)
        std = x.std(dim = [-2, -1], keepdim=True) + 1e-6
        x = (x - mean) / std
        H, W = x.shape[2], x.shape[3]
        
        # Calculate padding needed to make H and W divisible by 16
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        
        # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h)) 

        image_embedding = self.model(x)
        feature_embedding = self.feature_encoder(features)
        fused = torch.cat([image_embedding, feature_embedding], dim=1)
        
        return F.softplus(self.regression_head(fused).squeeze(-1))

class RelativeHuberLoss(nn.Module):
    def __init__(self, delta = 0.2, eps = 1e-8):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, pred, target):
        ratio = pred / (target + self.eps)
        target_ratio = torch.ones_like(ratio)
        return F.huber_loss(ratio, target_ratio, delta=self.delta, reduction='mean')
    
    



