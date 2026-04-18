import os
import pandas as pd
import random
from pathlib import Path

# Define paths
drone_dir = Path(r"data/datasets/University-Release/train/drone")
output_csv = Path(r"data/gps_labels.csv")

def generate_random_gps():
    if not os.path.exists(drone_dir):
        print(f"Error: Path {drone_dir} not found. Check your dataset path.")
        return

    # Grab every folder name inside train/drone
    location_ids = [d for d in os.listdir(drone_dir) if os.path.isdir(os.path.join(drone_dir, d))]
    location_ids.sort()

    # Base coordinates (Using a central campus location)
    base_lat = 26.5123
    base_lon = 80.2329

    data = []
    for loc_id in location_ids:
        # Generate a random offset of roughly +/- 500 meters
        lat_offset = random.uniform(-0.005, 0.005)
        lon_offset = random.uniform(-0.005, 0.005)
        
        # Calculate new coordinate and round to 6 decimal places for standard GPS precision
        lat = round(base_lat + lat_offset, 6)
        lon = round(base_lon + lon_offset, 6)
        
        data.append({"location_id": loc_id, "latitude": lat, "longitude": lon})

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    
    print(f"Success! Generated {output_csv} containing {len(location_ids)} location IDs.")
    print(f"Coordinates randomized around base [{base_lat}, {base_lon}] with a ~500m radius.")

if __name__ == "__main__":
    generate_random_gps()