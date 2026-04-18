import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class CrossViewDataset(Dataset):
    def __init__(self, drone_dir, sat_dir, csv_file, uav_tf=None, sat_tf=None, valid_locs=None):
        self.drone_dir = Path(drone_dir)
        self.sat_dir = Path(sat_dir)
        self.uav_tf = uav_tf
        self.sat_tf = sat_tf
        
        # Load GPS coordinates
        df = pd.read_csv(csv_file)
        df['location_id'] = df['location_id'].astype(str).str.zfill(4)
        self.gps_map = df.set_index('location_id').to_dict('index')

        # Filter valid locations if a split array (e.g., train_locs.npy) is provided
        if valid_locs is not None:
            valid_locs = set(str(loc).zfill(4) for loc in valid_locs)
        
        # Map satellite images first
        self.sat_map = {path.parent.name: path for path in self.sat_dir.rglob('*.jpg')}
        
        # Map drone images, ensuring they exist in the valid split
        all_drone_images = list(self.drone_dir.rglob('*.jpeg'))
        self.drone_images = []
        
        for path in all_drone_images:
            loc_id = path.parent.name
            if loc_id in self.sat_map:
                if valid_locs is None or loc_id in valid_locs:
                    self.drone_images.append(path)
                    
        print(f"Dataset initialized: {len(self.drone_images)} drone images mapped for this split.")
        
    def __len__(self):
        return len(self.drone_images)

    def __getitem__(self, idx):
        drone_path = self.drone_images[idx]
        loc_id = drone_path.parent.name
        
        drone_img = Image.open(drone_path).convert('RGB')
        sat_img = Image.open(self.sat_map[loc_id]).convert('RGB')
            
        # Apply independent transforms
        if self.uav_tf:
            drone_img = self.uav_tf(drone_img)
        if self.sat_tf:
            sat_img = self.sat_tf(sat_img)
            
        coords = self.gps_map.get(loc_id, {'latitude': 0.0, 'longitude': 0.0})
        lat = torch.tensor(coords['latitude'], dtype=torch.float32)
        lon = torch.tensor(coords['longitude'], dtype=torch.float32)
        
        return drone_img, sat_img, loc_id, lat, lon