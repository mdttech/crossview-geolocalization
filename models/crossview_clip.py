import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

class CLIPEncoder(nn.Module): 
    def __init__(self, embed_dim=512): 
        super().__init__() 
        # Downloads and loads the OpenAI pretrained weights
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai') 
        self.visual = clip_model.visual 
        
        # ViT-B-16 outputs 512 dimensions naturally, we project it just in case
        self.proj = nn.Linear(512, embed_dim) 
        self.bn   = nn.BatchNorm1d(embed_dim) 
        
    def forward(self, x): 
        f = self.visual(x) 
        return F.normalize(self.bn(self.proj(f)), p=2, dim=1) 

class CrossViewCLIPModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # Dual branches, both initialized with the massive CLIP brain
        self.uav_enc = CLIPEncoder(embed_dim=embed_dim)
        self.sat_enc = CLIPEncoder(embed_dim=embed_dim)
        
    def forward(self, uav, sat):
        return self.uav_enc(uav), self.sat_enc(sat)