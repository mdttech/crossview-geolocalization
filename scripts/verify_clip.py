import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crossview_clip import CrossViewCLIPModel
from data.datasets.crossview_dataset import CrossViewDataset
from data.transforms import uav_transform, sat_transform

def verify_checkpoint():
    print("1. Initializing OpenAI CLIP Model... (This may take a minute to download weights)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossViewCLIPModel().to(device)
    model.eval()
    print("-> CLIP encoder loaded without error!")

    print("\n2. Loading Sample Images from University-1652...")
    project_root = Path(__file__).parent.parent
    drone_dir = project_root / "data" / "datasets" / "University-Release" / "train" / "drone"
    sat_dir = project_root / "data" / "datasets" / "University-Release" / "train" / "satellite"
    csv_file = project_root / "data" / "gps_labels.csv"
    train_locs = np.load(project_root / "data" / "splits" / "train_locs.npy")

    dataset = CrossViewDataset(
        drone_dir, sat_dir, csv_file, 
        uav_tf=uav_transform(224), sat_tf=sat_transform(224), valid_locs=train_locs
    )

    # Grab Image A (UAV) and its true match (Sat A)
    uav_A, sat_A, loc_id_A, _, _ = dataset[0]
    
    # Grab a totally different location (Sat B)
    # We use index 100 to ensure it's a completely different building/location
    _, sat_B, loc_id_B, _, _ = dataset[100]

    # Add batch dimension and move to GPU
    uav_A = uav_A.unsqueeze(0).to(device)
    sat_A = sat_A.unsqueeze(0).to(device)
    sat_B = sat_B.unsqueeze(0).to(device)

    print(f"\n3. Running Forward Pass (Bypassing Random Projection Heads)...")
    import torch.nn.functional as F
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Extract features directly from the pre-trained visual backbone!
            raw_uav_A = model.uav_enc.visual(uav_A)
            raw_sat_A = model.sat_enc.visual(sat_A)
            raw_sat_B = model.sat_enc.visual(sat_B)
            
            # L2 Normalize them for Cosine Similarity
            emb_uav_A = F.normalize(raw_uav_A, p=2, dim=1)
            emb_sat_A = F.normalize(raw_sat_A, p=2, dim=1)
            emb_sat_B = F.normalize(raw_sat_B, p=2, dim=1)
            
    # Calculate Cosine Similarities
    sim_match = (emb_uav_A @ emb_sat_A.T).item()
    sim_random = (emb_uav_A @ emb_sat_B.T).item()
            

    print("\n--- RESULTS ---")
    print(f"Location ID A (Target): {loc_id_A}")
    print(f"Location ID B (Random): {loc_id_B}")
    print(f"Similarity (UAV A <-> True Sat A):   {sim_match:.4f}")
    print(f"Similarity (UAV A <-> Random Sat B): {sim_random:.4f}")

    if sim_match > sim_random:
        print("\nCHECKPOINT PASSED: The matching satellite score is higher than the random satellite score!")
    else:
        print("\nCHECKPOINT FAILED: The random satellite scored higher. (This happens occasionally with zero-shot, run again to verify).")

if __name__ == "__main__":
    verify_checkpoint()