import os
import sys
import yaml
import csv
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crossview_vit import CrossViewViTModel
from models.losses import InfoNCELoss
from data.datasets.crossview_dataset import CrossViewDataset
from data.transforms import uav_transform, sat_transform

def load_config(config_path="configs/vit_infonce.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def set_module_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad

def adjust_freezing_phase(model, epoch):
    if epoch == 1:
        print("\n--- PHASE 1: Warm-up (Freezing ViT Backbone) ---")
        for enc in [model.uav_enc, model.sat_enc]:
            set_module_grad(enc.backbone, False)
            
    elif epoch == 6:
        print("\n--- PHASE 2: Partial Fine-Tuning (Unfreezing Blocks 8-11) ---")
        for enc in [model.uav_enc, model.sat_enc]:
            for i in range(8, 12):
                set_module_grad(enc.backbone.blocks[i], True)
            set_module_grad(enc.backbone.norm, True)
            
    elif epoch == 31:
        print("\n--- PHASE 3: Full Fine-Tuning (Unfreezing All Blocks) ---")
        for enc in [model.uav_enc, model.sat_enc]:
            set_module_grad(enc.backbone, True)

# --- THE FIX: True Location ID Evaluation ---
def evaluate_recall(model, val_loader, device, k=1):
    model.eval()
    uav_embs, sat_embs, all_labels = [], [], []
    
    with torch.no_grad():
        for uav, sat, loc_id, _, _ in val_loader:
            u, s = model(uav.to(device), sat.to(device))
            uav_embs.append(u.cpu())
            sat_embs.append(s.cpu())
            all_labels.extend(loc_id)
            
    uav_embs = torch.cat(uav_embs)
    sat_embs = torch.cat(sat_embs)
    
    sim = uav_embs @ sat_embs.T
    top_k_indices = sim.topk(k, dim=1)[1]
    
    correct = 0
    for i, row in enumerate(top_k_indices):
        true_label = all_labels[i]
        predicted_labels = [all_labels[idx] for idx in row.tolist()]
        if true_label in predicted_labels:
            correct += 1
            
    return (correct / len(all_labels)) * 100.0

def train_model():
    project_root = Path(__file__).parent.parent
    config = load_config(project_root / "configs" / "vit_infonce.yaml")
    
    drone_dir = project_root / "data" / "datasets" / "University-Release" / "train" / "drone"
    sat_dir = project_root / "data" / "datasets" / "University-Release" / "train" / "satellite"
    gps_csv_file = project_root / "data" / "gps_labels.csv"
    
    save_dir = project_root / "models" / "saved_weights"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = results_dir / f"training_log_vit.csv"
    
    if not metrics_log_path.exists():
        with open(metrics_log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Recall_1', 'Learning_Rate', 'Is_Best_Model'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_locs = np.load(project_root / "data" / "splits" / "train_locs.npy")
    val_locs = np.load(project_root / "data" / "splits" / "val_locs.npy")

    img_size = config['training']['img_size']
    train_dataset = CrossViewDataset(drone_dir, sat_dir, gps_csv_file, uav_tf=uav_transform(img_size), sat_tf=sat_transform(img_size), valid_locs=train_locs)
    val_dataset = CrossViewDataset(drone_dir, sat_dir, gps_csv_file, uav_tf=uav_transform(img_size), sat_tf=sat_transform(img_size), valid_locs=val_locs)

    # --- THE FIX: No Accumulation Steps ---
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    mae_path = project_root / config['model']['mae_checkpoint']
    model = CrossViewViTModel(embed_dim=config['model']['embed_dim'], mae_weight_path=str(mae_path)).to(device)
    criterion = InfoNCELoss(temperature=config['loss']['temperature']).to(device)
    
    # --- THE FIX: Proper Optimizer Split ---
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "proj" in name or "bn" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = AdamW([
        {'params': backbone_params, 'lr': float(config['optimizer']['lr_backbone'])},
        {'params': head_params, 'lr': float(config['optimizer']['lr_head'])}
    ], weight_decay=float(config['optimizer']['weight_decay']))
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config['scheduler']['T_max'])
    scaler = torch.amp.GradScaler('cuda') 

    epochs = config['training']['epochs']
    patience = config['training']['patience']
    best_r1 = 0.0
    epochs_without_improvement = 0
    start_epoch = 1

    # --- THE FIX: Auto-Resume Logic ---
    resume_path = save_dir / "best_vit_mae.pth"
    if resume_path.exists():
        print(f"[*] Found existing checkpoint! Resuming from: {resume_path.name}")
        checkpoint = torch.load(resume_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        best_r1 = checkpoint.get('r1', 0.0)
        start_epoch = checkpoint['epoch'] + 1
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        print(f"[*] Successfully restored Model weights. Resuming at Epoch {start_epoch} with Best R@1: {best_r1:.2f}%")
    else:
        print("[*] No checkpoint found. Starting training from scratch.")

    print(f"\n Starting ViT Training with MAE Initialization. Metrics logging to: {metrics_log_path}")

    for epoch in range(start_epoch, epochs + 1):
        adjust_freezing_phase(model, epoch)

        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for i, (uav, sat, loc_id, lat, lon) in enumerate(pbar):
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                u_emb, s_emb = model(uav.to(device), sat.to(device))
                loss = criterion(u_emb, s_emb)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        r1 = evaluate_recall(model, val_loader, device, k=1)
        
        print(f"-> Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Recall@1: {r1:.2f}% | LR: {current_lr:.2e}")

        is_best = False
        if r1 > best_r1:
            best_r1 = r1
            is_best = True
            epochs_without_improvement = 0
            best_path = save_dir / f"best_vit_mae.pth"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'r1': best_r1, 'config': config, 'optimizer_state_dict': optimizer.state_dict()}, best_path)
            print(f"*** New Best ViT Model! Recall@1: {best_r1:.2f}% ***")
        else:
            epochs_without_improvement += 1

        with open(metrics_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_train_loss:.4f}", f"{r1:.2f}", f"{current_lr:.6f}", is_best])

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {patience} epochs.")
            break

if __name__ == "__main__":
    train_model()