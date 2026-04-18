import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent
drone_dir = project_root / "data" / "datasets" / "University-Release" / "train" / "drone"
# Create a folder to hold rejected images so they are skipped by the dataloader
rejected_dir = project_root / "data" / "datasets" / "University-Release" / "train" / "rejected_drone"

def is_blurry(path, threshold=80.0):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: 
        return True
    return cv2.Laplacian(img, cv2.CV_64F).var() < threshold

def run_pipeline():
    rejected_dir.mkdir(exist_ok=True)
    images = list(drone_dir.rglob('*.jpeg'))
    blurry_count = 0
    
    print(f"Scanning {len(images)} UAV images for blur (Threshold: 80.0)...")
    
    for img_path in tqdm(images, desc="Filtering Blurry Images"):
        if is_blurry(img_path):
            blurry_count += 1
            # Move the blurry file out of the training directory
            new_name = f"{img_path.parent.name}_{img_path.name}"
            shutil.move(str(img_path), str(rejected_dir / new_name))
            
    print("\n--- Preprocessing Report ---")
    print(f"Total UAV Images Scanned: {len(images)}")
    print(f"Images identified as blurry and moved to rejection folder: {blurry_count}")
    print(f"Clean, usable UAV images remaining in dataset: {len(images) - blurry_count}")
    print("Checkpoint Passed: End-to-End preprocessing executed.")

if __name__ == "__main__":
    run_pipeline()