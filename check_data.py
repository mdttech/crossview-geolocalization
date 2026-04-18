import os
from pathlib import Path

# Define paths based on your exact structure
drone_dir = Path(r"data/datasets/University-Release/train/drone")
sat_dir = Path(r"data/datasets/University-Release/train/satellite")

def verify_dataset():
    print("--- Dataset Verification Report ---\n")
    
    # 1. Check Drone Data
    if drone_dir.exists():
        drone_folders = [d for d in drone_dir.iterdir() if d.is_dir()]
        drone_images = list(drone_dir.rglob('*.jpeg'))
        print(f"DRONE DATA:")
        print(f"  - Subfolders found (e.g., 0839 to 1650): {len(drone_folders)}")
        print(f"  - Total .jpeg images found: {len(drone_images)}")
    else:
        print(f"DRONE DATA: Directory not found at {drone_dir}")

    print("-" * 35)
    
    # 2. Check Satellite Data
    if sat_dir.exists():
        sat_folders = [d for d in sat_dir.iterdir() if d.is_dir()]
        sat_images = list(sat_dir.rglob('*.jpg'))
        print(f"SATELLITE DATA:")
        print(f"  - Subfolders found (e.g., 0839 to 1650): {len(sat_folders)}")
        print(f"  - Total .jpg images found: {len(sat_images)}")
    else:
        print(f"SATELLITE DATA: Directory not found at {sat_dir}")

if __name__ == "__main__":
    verify_dataset()