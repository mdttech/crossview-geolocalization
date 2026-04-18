"""
Fully Automated Greedy Ablation Study.
Runs: python scripts/run_ablation.py
"""

import os, sys, json, time
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.datasets.crossview_dataset import CrossViewDataset
from data.transforms import uav_transform, sat_transform

SEEDS = [42]
RESULTS_CSV = "results/ablation_results.csv"

@dataclass
class AblationConfig:
    backbone: str = "vit_base_patch16_224"
    loss: str = "infonce"
    pooling: str = "gem_p3"
    embed_dim: int = 512
    aug_strategy: str = "asymmetric"
    seed: int = 42
    epochs: int = 30
    batch_size: int = 32 # Set to 32 to ensure ViT-B fits in Low VRAM
    img_size: int = 224
    lr_backbone: float = 5e-5
    lr_head: float = 5e-4
    weight_decay: float = 0.01
    temperature: float = 0.07
    mae_weights: str = "models/saved_weights/mae_satellite_best.pth" 

def build_model(cfg: AblationConfig, device):
    import torch.nn as nn
    import torch.nn.functional as F
    import timm, torchvision.models as tvm
    from torchvision.models import ResNet50_Weights

    class GeM(nn.Module):
        def __init__(self, p=3.0, tunable=False):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1)*p) if tunable else p
        def forward(self, x):
            return F.adaptive_avg_pool2d(x.clamp(1e-6).pow(self.p), 1).pow(1.0/self.p)

    def make_pool(name):
        if name == "gap": return nn.AdaptiveAvgPool2d(1)
        if name == "gem_p3": return GeM(p=3.0, tunable=False)
        if name == "gem_tuned": return GeM(p=3.0, tunable=True)
        raise ValueError(name)

    class ViTEncoderFull(nn.Module):
        def __init__(self, model_name, embed_dim, mae_weights=None):
            super().__init__()
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
            if mae_weights and Path(mae_weights).exists():
                sd = {k.replace("encoder.", ""): v for k, v in torch.load(mae_weights, map_location="cpu").items() if "encoder" in k}
                self.backbone.load_state_dict(sd, strict=False)
            self.proj = nn.Linear(self.backbone.num_features, embed_dim)
            self.bn   = nn.BatchNorm1d(embed_dim)

        def forward(self, x):
            return F.normalize(self.bn(self.proj(self.backbone(x))), p=2, dim=1)

    class CLIPEncoderFull(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            import open_clip
            clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
            self.visual = clip_model.visual
            self.proj   = nn.Linear(512, embed_dim)
            self.bn     = nn.BatchNorm1d(embed_dim)
        def forward(self, x):
            return F.normalize(self.bn(self.proj(self.visual(x))), p=2, dim=1)

    class ResNetEncoderFull(nn.Module):
        def __init__(self, embed_dim, pool_name):
            super().__init__()
            bb = tvm.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
            self.layers = nn.Sequential(bb.layer1, bb.layer2, bb.layer3, bb.layer4)
            self.pool = make_pool(pool_name)
            self.proj = nn.Linear(2048, embed_dim)
            self.bn   = nn.BatchNorm1d(embed_dim)
        def forward(self, x):
            feat = self.pool(self.layers(self.stem(x))).flatten(1)
            return F.normalize(self.bn(self.proj(feat)), p=2, dim=1)

    class CrossViewModel(nn.Module):
        def __init__(self, u, s): super().__init__(); self.uav_enc = u; self.sat_enc = s
        def forward(self, u, s): return self.uav_enc(u), self.sat_enc(s)
        def encode_uav(self, x): return self.uav_enc(x)
        def encode_sat(self, x): return self.sat_enc(x)

    d, p, mw = cfg.embed_dim, cfg.pooling, cfg.mae_weights
    
    if cfg.backbone == "resnet50":
        u_enc, s_enc = ResNetEncoderFull(d, p), ResNetEncoderFull(d, p)
    elif cfg.backbone.startswith("vit"):
        u_enc, s_enc = ViTEncoderFull(cfg.backbone, d), ViTEncoderFull(cfg.backbone, d, mw)
    elif cfg.backbone == "clip_vit_b16":
        u_enc, s_enc = CLIPEncoderFull(d), CLIPEncoderFull(d)

    model = CrossViewModel(u_enc, s_enc).to(device)

    # --- Split Learning Rates for Ablation ---
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "proj" in name or "bn" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg.lr_backbone},
        {'params': head_params, 'lr': cfg.lr_head}
    ], weight_decay=cfg.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    return model, optimizer, scheduler

def build_loss(cfg: AblationConfig):
    import torch.nn as nn
    import torch.nn.functional as F

    # 1. Standard InfoNCE
    class InfoNCE(nn.Module):
        def __init__(self, tau): 
            super().__init__()
            self.tau = tau
            
        def forward(self, u, s):
            logits = torch.mm(u, s.T) / self.tau
            labels = torch.arange(u.size(0), device=u.device)
            return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    # 2. Triplet Loss (Vectorized with Hard Negative Mining)
    class TripletLoss(nn.Module):
        def __init__(self, margin=0.3): 
            super().__init__()
            self.margin = margin
            
        def forward(self, u, s):
            # Calculate all pairwise distances in the batch
            distance_matrix = torch.cdist(u, s, p=2)
            
            # Positives are the matching images on the diagonal
            positives = distance_matrix.diag()
            
            # Mask out the diagonal so we don't pick the true match as a negative
            mask = torch.eye(u.size(0), device=u.device).bool()
            distance_matrix = distance_matrix.masked_fill(mask, float('inf'))
            
            # Find the "Hardest Negative" (the wrong satellite image that looks most like the drone image)
            negatives = distance_matrix.min(dim=1).values
            
            # Standard Triplet Math: max(0, PositiveDistance - NegativeDistance + Margin)
            loss = F.relu(positives - negatives + self.margin)
            return loss.mean()

    # 3. InfoNCE + Auxiliary Classification (Proxy)
    class InfoNCEWithCLS(nn.Module):
        def __init__(self, tau, cls_weight=0.5): 
            super().__init__()
            self.tau = tau
            self.cls_weight = cls_weight
            
        def forward(self, u, s):
            # Calculate standard InfoNCE
            logits = torch.mm(u, s.T) / self.tau
            labels = torch.arange(u.size(0), device=u.device)
            infonce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            
            # Proxy CLS Loss: We add an extra penalty if the model can't distinctively classify 
            # the drone images from EACH OTHER in the batch space.
            cls_loss = F.cross_entropy(logits, labels) 
            
            return infonce + (self.cls_weight * cls_loss)

    # --- Routing Logic ---
    if cfg.loss == "infonce":
        return InfoNCE(cfg.temperature)
    elif cfg.loss == "triplet":
        return TripletLoss()
    elif cfg.loss == "infonce+cls":
        return InfoNCEWithCLS(cfg.temperature)
    else:
        raise ValueError(f"Unknown loss function requested: {cfg.loss}")

def run_single_experiment(cfg: AblationConfig, train_loader, val_loader, device):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model, optimizer, scheduler = build_model(cfg, device)
    criterion = build_loss(cfg)
    scaler = torch.amp.GradScaler('cuda')
    best_r1 = 0.0

    print(f"--> Training: {cfg.backbone} | Loss: {cfg.loss} | Pool: {cfg.pooling} | Dim: {cfg.embed_dim} | Seed: {cfg.seed}")
    
    for epoch in range(cfg.epochs):
        model.train()
        for uav, sat, _, _, _ in train_loader:
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                u_emb, s_emb = model(uav.to(device), sat.to(device))
                loss = criterion(u_emb, s_emb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Eval
        model.eval()
        uavs, sats, labels = [], [], []
        with torch.no_grad():
            for uav, sat, lbl, _, _ in val_loader:
                uavs.append(model.encode_uav(uav.to(device)).cpu())
                sats.append(model.encode_sat(sat.to(device)).cpu())
                labels.extend(lbl)
        
        sim = torch.mm(torch.cat(uavs), torch.cat(sats).T)
        top_k = sim.topk(1, dim=1).indices
        correct = sum(1 for i, row in enumerate(top_k) if labels[i] == labels[row[0]])
        r1 = (correct / len(labels)) * 100
        
        if r1 > best_r1: best_r1 = r1

    return best_r1

def execute_tournament(device):
    Path("results").mkdir(exist_ok=True)
    results = []
    
    # --- RESUME LOGIC ---
    completed_runs = {}
    if Path(RESULTS_CSV).exists():
        df_existing = pd.read_csv(RESULTS_CSV)
        for _, row in df_existing.iterrows():
            # Create a unique key for each run based on its config
            key = f"{row['backbone']}_{row['loss']}_{row['pooling']}_{row['embed_dim']}_{row['aug_strategy']}_{row['seed']}"
            completed_runs[key] = row.to_dict()
        results.extend(df_existing.to_dict('records'))
        print(f"[*] Found {len(completed_runs)} completed runs in CSV. Will skip these.")
    # --------------------------------

    project_root = Path(__file__).parent.parent
    train_locs = np.load(project_root / "data" / "splits" / "train_locs.npy")
    val_locs = np.load(project_root / "data" / "splits" / "val_locs.npy")
    dataset_args = (project_root/"data"/"datasets"/"University-Release"/"train"/"drone", project_root/"data"/"datasets"/"University-Release"/"train"/"satellite", project_root/"data"/"gps_labels.csv")
    
    # Base loaders
    train_loader = torch.utils.data.DataLoader(CrossViewDataset(*dataset_args, uav_tf=uav_transform(224), sat_tf=sat_transform(224), valid_locs=train_locs), batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(CrossViewDataset(*dataset_args, uav_tf=uav_transform(224), sat_tf=sat_transform(224), valid_locs=val_locs), batch_size=32, shuffle=False, num_workers=4)

    def run_round(configs, round_name):
        print(f"\n{'='*50}\nSTARTING {round_name}\n{'='*50}")
        round_results = []
        for c in configs:
            for s in SEEDS:
                # --- CHECK IF RUN ALREADY EXISTS ---
                key = f"{c['backbone']}_{c['loss']}_{c['pooling']}_{c['embed_dim']}_{c['aug_strategy']}_{s}"
                if key in completed_runs:
                    print(f"    [Skipping] Already completed: {c['backbone']} | {c['loss']} | {c['pooling']} | Dim: {c['embed_dim']}")
                    row = completed_runs[key]
                    round_results.append(row)
                    continue
                # -----------------------------------
                
                cfg = AblationConfig(**c, seed=s)
                t0 = time.time()
                r1 = run_single_experiment(cfg, train_loader, val_loader, device)
                mins = round((time.time() - t0) / 60, 1)
                
                row = dict(c); row.update({"seed": s, "best_r1": r1, "minutes": mins, "round": round_name})
                results.append(row)
                round_results.append(row)
                pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
                completed_runs[key] = row # Add to memory so it doesn't repeat
                print(f"    R@1 = {r1:.2f}% ({mins} min)")
                
        # Calculate winner of the round
        df = pd.DataFrame(round_results)
        winner = df.groupby(list(c.keys()))['best_r1'].mean().idxmax()
        return dict(zip(c.keys(), winner))

    # --- THE TOURNAMENT PIPELINE ---
    
    # ROUND 1: Backbones
    r1_configs = [
        {"backbone": "resnet50", "loss": "infonce", "pooling": "gem_p3", "embed_dim": 512, "aug_strategy": "asymmetric"},
        {"backbone": "vit_small_patch16_224", "loss": "infonce", "pooling": "gem_p3", "embed_dim": 512, "aug_strategy": "asymmetric"},
        {"backbone": "vit_base_patch16_224", "loss": "infonce", "pooling": "gem_p3", "embed_dim": 512, "aug_strategy": "asymmetric"}
    ]
    best_r1 = run_round(r1_configs, "ROUND 1: Backbones")
    best_bb = best_r1["backbone"]
    print(f"\nROUND 1 WINNER: {best_bb}\n")

    # ROUND 2: Loss
    r2_configs = [
        {"backbone": best_bb, "loss": "infonce", "pooling": "gem_p3", "embed_dim": 512, "aug_strategy": "asymmetric"},
        {"backbone": best_bb, "loss": "triplet", "pooling": "gem_p3", "embed_dim": 512, "aug_strategy": "asymmetric"}
    ]
    best_r2 = run_round(r2_configs, "ROUND 2: Loss")
    best_loss = best_r2["loss"]
    print(f"\nROUND 2 WINNER: {best_loss}\n")

    # ROUND 3: Pooling & Dimensions
    r3_configs = [
        {"backbone": best_bb, "loss": best_loss, "pooling": "gem_p3", "embed_dim": 512, "aug_strategy": "asymmetric"},
        {"backbone": best_bb, "loss": best_loss, "pooling": "gem_tuned", "embed_dim": 512, "aug_strategy": "asymmetric"},
        {"backbone": best_bb, "loss": best_loss, "pooling": "gem_p3", "embed_dim": 1024, "aug_strategy": "asymmetric"},
        {"backbone": best_bb, "loss": best_loss, "pooling": "gem_p3", "embed_dim": 256, "aug_strategy": "asymmetric"}
    ]
    best_r3 = run_round(r3_configs, "ROUND 3: Pooling & Dims")
    best_pool, best_dim = best_r3["pooling"], best_r3["embed_dim"]
    print(f"\nROUND 3 WINNER: Pool={best_pool}, Dim={best_dim}\n")

    # ROUND 5: CLIP Baseline (Optional Bonus)
    r5_configs = [
        {"backbone": "clip_vit_b16", "loss": best_loss, "pooling": best_pool, "embed_dim": best_dim, "aug_strategy": "asymmetric"}
    ]
    run_round(r5_configs, "ROUND 5: CLIP Baseline")

    print(f"\nTOURNAMENT COMPLETE! All results saved to {RESULTS_CSV}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    execute_tournament(device)