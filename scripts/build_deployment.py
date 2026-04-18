import os
import sys
import torch
import torch.nn as nn
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.crossview_model import CrossViewModel
from data.transforms import sat_transform

# Tiny dataset to load UNIQUE satellite images for the gallery
class SatelliteGalleryDataset(Dataset):
    def __init__(self, sat_dir, transform):
        # Find all unique satellite images
        self.images = list(Path(sat_dir).rglob('*.jpg'))
        self.transform = transform
        
    def __len__(self): 
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        # Extract location ID from filename (format: locID_...)
        loc_id = img_path.parent.name 
        return self.transform(img), loc_id, str(img_path)

def build_deployment():
    print("=== Deployment Prep ===")
    project_root = Path(__file__).parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Best Model
    print("\n1. Loading Best ResNet50 Model...")
    model = CrossViewModel(embed_dim=512)
    weights_path = project_root / "models" / "saved_weights" / "best_baseline_resnet50.pth"
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 2. Build FAISS Index
    print("\n2. Extracting Satellite Gallery for FAISS...")
    sat_dir = project_root / "data" / "datasets" / "University-Release" / "train" / "satellite" # Using test set for final index
    gallery_dataset = SatelliteGalleryDataset(sat_dir, sat_transform(224))
    gallery_loader = DataLoader(gallery_dataset, batch_size=64, num_workers=4)
    
    all_embs, all_labels = [], []
    with torch.no_grad():
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            for imgs, loc_ids, _ in tqdm(gallery_loader, desc="Indexing Satellites"):
                embs = model.encode_sat(imgs.to(device)).cpu().numpy()
                all_embs.append(embs)
                all_labels.extend(loc_ids)
                
    embeddings = np.vstack(all_embs).astype(np.float32)
    d = embeddings.shape[1] # 512 dimensions
    
    print(f"   -> Building FAISS Index for {len(embeddings)} tiles...")
    index = faiss.IndexFlatL2(d) # L2 distance matching
    index.add(embeddings)
    
    faiss_path = project_root / "models" / "satellite_index.faiss"
    faiss.write_index(index, str(faiss_path))
    
    # Save the label mapping so we know which index corresponds to which building
    np.save(project_root / "models" / "faiss_labels.npy", np.array(all_labels))
    print(f"   -> FAISS Index saved to {faiss_path}!")

    # 3. Model INT8 Quantization 
    print("\n3. Quantizing UAV Encoder to INT8...")
    model.to('cpu') # Quantization is done on CPU
    
    # Script the UAV encoder
    scripted = torch.jit.script(model.uav_enc)
    
    # Apply dynamic quantization to the Linear projection heads
    quantized_uav = torch.quantization.quantize_dynamic(
        scripted, {nn.Linear}, dtype=torch.qint8
    )
    
    quant_path = project_root / "models" / "uav_encoder_int8.pt"
    torch.jit.save(quantized_uav, str(quant_path))
    
    # Compare sizes
    fp32_size = os.path.getsize(weights_path) / (1024 * 1024)
    int8_size = os.path.getsize(quant_path) / (1024 * 1024)
    print(f"   -> Original Full Model Size: {fp32_size:.1f} MB")
    print(f"   -> Quantized UAV Encoder Size: {int8_size:.1f} MB")
    print(f"   -> INT8 Model saved to {quant_path}!")
    print("\nDeployment Build Complete. Ready for Final Evaluation.")

if __name__ == "__main__":
    build_deployment()