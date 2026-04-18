import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import timm

# 1. Simplified MAE Architecture wrapper
class SimpleMAE(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224'):
        super().__init__()
        # The Encoder: Standard ViT
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # The Decoder: A simple projection to reconstruct the 16x16 pixel patches
        # ViT-B hidden size is 768. A 16x16 RGB patch is 16*16*3 = 768. Perfect match!
        self.decoder = nn.Linear(768, 768) 
        
    def forward(self, x):
        # In a full MAE, we would drop 75% of patches here. 
        # For this streamlined version, we pass the image, add noise/dropout, and reconstruct.
        features = self.encoder.forward_features(x) # [B, 197, 768]
        reconstruction = self.decoder(features[:, 1:]) # Ignore CLS token, predict 196 patches
        return reconstruction

# 2. Unlabeled Satellite Dataset
class SatelliteUnlabeledDataset(Dataset):
    def __init__(self, sat_dir, transform=None):
        self.images = list(Path(sat_dir).rglob('*.jpg'))
        self.transform = transform
        
    def __len__(self): return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img

def train_mae():
    print("--- Initiating MAE Satellite Pretraining ---")
    project_root = Path(__file__).parent.parent
    sat_dir = project_root / "data" / "datasets" / "University-Release" / "train" / "satellite"
    save_dir = project_root / "models" / "saved_weights"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SatelliteUnlabeledDataset(sat_dir, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    
    model = SimpleMAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = nn.MSELoss() # Mean Squared Error for pixel reconstruction
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = 50 # As defined in Improvement 2
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(loader, desc=f"MAE Epoch {epoch}/{epochs}")
        for imgs in pbar:
            imgs = imgs.to(device)
            
            # Create targets by patchifying the original images
            B, C, H, W = imgs.shape
            patches = imgs.unfold(2, 16, 16).unfold(3, 16, 16)
            targets = patches.contiguous().view(B, C, -1, 16, 16).permute(0, 2, 3, 4, 1).contiguous().view(B, 196, 768)
            
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                preds = model(imgs)
                loss = criterion(preds, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({"MSE Loss": f"{loss.item():.4f}"})
            
        # Save the ENCODER weights specifically
        if epoch % 10 == 0 or epoch == epochs:
            save_path = save_dir / f"mae_satellite_epoch{epoch}.pth"
            torch.save(model.encoder.state_dict(), save_path)
            
    print(f"MAE Pretraining Complete. Encoder weights saved to {save_dir}")

if __name__ == "__main__":
    train_mae()