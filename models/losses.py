import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature
        
    def forward(self, uav_emb, sat_emb):
        B = uav_emb.size(0)
        # Compute similarity matrix and scale by temperature
        logits = torch.mm(uav_emb, sat_emb.T) / self.tau  # [B, B]
        
        # The correct matches are on the diagonal (0 to B-1)
        labels = torch.arange(B, device=uav_emb.device)
        
        # Symmetric cross entropy (UAV->Sat and Sat->UAV)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        
        # Soft-margin triplet loss
        return F.relu(d_pos - d_neg + self.margin).mean()

# --- Checkpoint Verification ---
if __name__ == "__main__":
    print("Testing Loss Functions...")
    
    # 1. Initialize InfoNCE
    criterion = InfoNCELoss(temperature=1.0) # Set tau=1.0 strictly to test the log(8) math expectation
    
    # 2. Simulate a batch of 8 embeddings (Dimension: 512)
    B = 8
    embed_dim = 512
    
    # Create purely orthogonal/random vectors to simulate an untrained state
    dummy_uav = F.normalize(torch.randn(B, embed_dim), p=2, dim=1)
    dummy_sat = F.normalize(torch.randn(B, embed_dim), p=2, dim=1)
    
    # 3. Calculate Loss
    loss_val = criterion(dummy_uav, dummy_sat)
    
    # 4. Math verification: Untrained loss of B=8 should approach ln(8) ≈ 2.079
    import math
    expected_loss = math.log(B)
    
    print(f"\n--- InfoNCE Verification ---")
    print(f"Batch Size: {B}")
    print(f"Expected theoretical loss: ~{expected_loss:.4f}")
    print(f"Actual computed loss:      {loss_val.item():.4f}")
    
    if abs(loss_val.item() - expected_loss) < 0.5:
         print("\nCheckpoint Passed: InfoNCE is computing correctly!")
    else:
         print("\nWarning: Loss value deviates significantly from expected math.")