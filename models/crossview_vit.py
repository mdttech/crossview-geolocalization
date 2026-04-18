import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path

class ViTEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', embed_dim=512, mae_weight_path=None):
        super().__init__()
        
        # 1. Initialize standard ViT
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # 2. Improvement: Inject custom MAE weights if provided
        if mae_weight_path and Path(mae_weight_path).exists():
            print(f"[*] Loading Satellite MAE Pretrained Weights from {mae_weight_path}...")
            # Strict=False allows us to load the encoder weights smoothly
            self.backbone.load_state_dict(torch.load(mae_weight_path), strict=False)
            print("[*] MAE Weights Successfully Injected!")
        else:
            print("[!] No MAE weights found. Defaulting to standard ImageNet ViT weights.")
            
        # 3. Projection Head (ViT-B outputs 768 dimensions)
        self.proj = nn.Linear(self.backbone.num_features, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        
    def forward(self, x):
        f = self.backbone(x)  # Extract CLS token [B, 768]
        f = self.bn(self.proj(f))
        return F.normalize(f, p=2, dim=1) # L2 Normalize

class CrossViewViTModel(nn.Module):
    def __init__(self, embed_dim=512, mae_weight_path=None):
        super().__init__()
        # Both branches use the ViT, but the satellite branch loads the MAE weights
        self.uav_enc = ViTEncoder(embed_dim=embed_dim)
        self.sat_enc = ViTEncoder(embed_dim=embed_dim, mae_weight_path=mae_weight_path)
        
    def forward(self, uav, sat):
        return self.uav_enc(uav), self.sat_enc(sat)

# --- Checkpoint Verification ---
if __name__ == "__main__":
    print("Testing Dual-Branch ViT Model...")
    model = CrossViewViTModel(embed_dim=512)
    model.eval()
    
    dummy_uav = torch.randn(2, 3, 224, 224)
    dummy_sat = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        uav_emb, sat_emb = model(dummy_uav, dummy_sat)
        
    assert uav_emb.shape == (2, 512), "UAV shape mismatch!"
    assert sat_emb.shape == (2, 512), "SAT shape mismatch!"
    print(f"Checkpoint Passed: Output shapes are {uav_emb.shape}. Ready for Phase 3 Training!")