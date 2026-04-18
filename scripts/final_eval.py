import os
import sys
import torch
import faiss
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.datasets.crossview_dataset import CrossViewDataset
from data.transforms import uav_transform, sat_transform
from evaluation.metrics import recall_at_k, mean_average_precision

def run_final_evaluation():
    print("=== Final Evaluation ===")
    project_root = Path(__file__).parent.parent
    device = torch.device("cpu") # Forcing CPU to test the INT8 deployment model
    
    # 1. Load INT8 Model & FAISS Index
    print("Loading INT8 UAV Encoder and FAISS Index...")
    uav_encoder = torch.jit.load(project_root / "models" / "uav_encoder_int8.pt", map_location='cpu')
    uav_encoder.eval()
    
    index = faiss.read_index(str(project_root / "models" / "satellite_index.faiss"))
    gallery_labels = np.load(project_root / "models" / "faiss_labels.npy")
    
    # 2. Load Drone Queries (Test Set)
    val_locs = np.load(project_root / "data" / "splits" / "val_locs.npy") # Or test_locs if you have them
    dataset_args = (project_root/"data"/"datasets"/"University-Release"/"train"/"drone", project_root/"data"/"datasets"/"University-Release"/"train"/"satellite", project_root/"data"/"gps_labels.csv")
    query_dataset = CrossViewDataset(*dataset_args, uav_tf=uav_transform(224), sat_tf=sat_transform(224), valid_locs=val_locs)
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    query_labels = []
    retrieved_per_query = []
    search_times = []
    
    print("Running Drone Queries through INT8 Encoder...")
    with torch.no_grad():
        for uav, _, loc_id, _, _ in tqdm(query_loader, desc="Querying"):
            # Extract 512D Vector
            q_embs = uav_encoder(uav).numpy()
            query_labels.extend(loc_id)
            
            # Search FAISS Index
            start_time = time.time()
            distances, indices = index.search(q_embs, k=10) # Grab top 10 matches
            search_times.append((time.time() - start_time) * 1000) # ms
            
            # Map FAISS indices back to actual building IDs
            for row in indices:
                retrieved_per_query.append([gallery_labels[idx] for idx in row])
                
    # 3. Calculate Metrics
    print("\n=== FINAL RESULTS ===")
    
    # Speed Metric
    avg_search_time = np.mean(search_times)
    print(f"Average FAISS Search Time: {avg_search_time:.2f} ms")
    assert avg_search_time < 10.0, "Checkpoint Failed: Search took longer than 10ms!"
    print(" Checkpoint Passed: Search completes in < 10 ms.")
    
    # Recall and mAP
    recalls = recall_at_k(query_labels, retrieved_per_query, k_values=[1, 5, 10])
    map_score = mean_average_precision(query_labels, retrieved_per_query)
    
    print(f"\nRecall@1:  {recalls[1]:.2f}%")
    print(f"Recall@5:  {recalls[5]:.2f}%")
    print(f"Recall@10: {recalls[10]:.2f}%")
    print(f"mAP:       {map_score:.2f}%")
    
    # If Recall@1 is within ~2% of 89% FP32 score, quantization was a success!

if __name__ == "__main__":
    run_final_evaluation()