import os
import numpy as np
from pathlib import Path

# Paths based on your structure
project_root = Path(__file__).parent.parent
drone_dir = project_root / "data" / "datasets" / "University-Release" / "train" / "drone"
splits_dir = project_root / "data" / "splits"

def generate_splits():
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Grab the 701 location IDs
    locs = np.array(sorted([d.name for d in drone_dir.iterdir() if d.is_dir()]))
    
    # Set seed for strict reproducibility
    np.random.seed(42)
    np.random.shuffle(locs)

    n = len(locs)
    
    # 70% Train, 10% Val, 20% Test
    train_locs = locs[:int(0.7 * n)]
    val_locs = locs[int(0.7 * n):int(0.8 * n)]
    test_locs = locs[int(0.8 * n):]

    # Save splits
    np.save(splits_dir / 'train_locs.npy', train_locs)
    np.save(splits_dir / 'val_locs.npy', val_locs)
    np.save(splits_dir / 'test_locs.npy', test_locs)

    # Verification checkpoint
    assert len(set(train_locs) & set(val_locs)) == 0, "Leakage between train and val!"
    assert len(set(val_locs) & set(test_locs)) == 0, "Leakage between val and test!"
    
    print(f"Splits generated successfully and locked in!")
    print(f"Train: {len(train_locs)} | Val: {len(val_locs)} | Test: {len(test_locs)}")

if __name__ == "__main__":
    generate_splits()