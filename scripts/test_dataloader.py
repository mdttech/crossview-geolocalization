import sys
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Ensure Python can find local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets.crossview_dataset import CrossViewDataset
from data.transforms import uav_transform, sat_transform

def verify_dataloader():
    project_root = Path(__file__).parent.parent
    drone_dir = project_root / "data" / "datasets" / "University-Release" / "train" / "drone"
    sat_dir = project_root / "data" / "datasets" / "University-Release" / "train" / "satellite"
    csv_file = project_root / "data" / "gps_labels.csv"
    
    # 1. Load the immutable 70% training split
    train_locs = np.load(project_root / "data" / "splits" / "train_locs.npy")
    print(f"Loaded {len(train_locs)} training locations from split.")

    # 2. Initialize the dataset with independent transforms
    dataset = CrossViewDataset(
        drone_dir=drone_dir,
        sat_dir=sat_dir,
        csv_file=csv_file,
        uav_tf=uav_transform(img_size=224),
        sat_tf=sat_transform(img_size=224),
        valid_locs=train_locs
    )

    # 3. Create the DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )

    print("Fetching a batch to verify dimensions (this will test the multiprocessor workers)...")
    
    # 4. Fetch exactly one batch to pass the checkpoint
    for uav, sat, loc_id, lat, lon in loader:
        print("\n--- Checkpoint Verification ---")
        print(f"UAV Batch Shape: {uav.shape} (Expected: [64, 3, 224, 224])")
        print(f"SAT Batch Shape: {sat.shape} (Expected: [64, 3, 224, 224])")
        
        assert uav.shape == (64, 3, 224, 224), "UAV shape mismatch!"
        assert sat.shape == (64, 3, 224, 224), "SAT shape mismatch!"
        
        print("Checkpoint Passed: DataLoader produces correct batches without errors.")
        break 

if __name__ == '__main__':
    verify_dataloader()