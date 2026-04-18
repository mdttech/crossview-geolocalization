import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class GeM(nn.Module):
    """Generalized Mean Pooling"""
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(self.p), 1).pow(1.0/self.p)

class ResNetFPNEncoder(nn.Module):
    def __init__(self, embed_dim=512, is_uav=False):
        super().__init__()
        self.is_uav = is_uav
        
        # Load Pretrained ResNet50 Backbone
        bb = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Separate the layers so we can extract multi-scale features
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1  
        self.layer2 = bb.layer2  
        self.layer3 = bb.layer3  
        self.layer4 = bb.layer4  
        
        # --- Instance Normalization (UAV Only) ---
        # affine=True allows the layer to learn optimal scaling after stripping style
        if self.is_uav:
            self.in1 = nn.InstanceNorm2d(256, affine=True)
            self.in2 = nn.InstanceNorm2d(512, affine=True)
            self.in3 = nn.InstanceNorm2d(1024, affine=True)
            
        # --- Feature Pyramid Network (FPN) 1x1 Convs ---
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1)
        self.latlayer2 = nn.Conv2d(512, 512, kernel_size=1)
        self.latlayer3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.latlayer4 = nn.Conv2d(2048, 512, kernel_size=1)
        
        self.pool = GeM(p=3.0)
        
        # We concatenate 4 scales of 512 channels = 2048
        self.proj = nn.Linear(2048, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        
    def forward(self, x):
        # Bottom-up pathway with Instance Normalization injections
        x = self.stem(x)
        
        c1 = self.layer1(x)
        if self.is_uav: c1 = self.in1(c1)
            
        c2 = self.layer2(c1)
        if self.is_uav: c2 = self.in2(c2)
            
        c3 = self.layer3(c2)
        if self.is_uav: c3 = self.in3(c3)
            
        c4 = self.layer4(c3)

        # Top-down FPN pathway
        p4 = self.latlayer4(c4)
        p3 = self.latlayer3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.latlayer2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.latlayer1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False)

        # GeM pool all multiple scales
        v4 = self.pool(p4).flatten(1)
        v3 = self.pool(p3).flatten(1)
        v2 = self.pool(p2).flatten(1)
        v1 = self.pool(p1).flatten(1)

        # Combine multi-scale features [B, 512 * 4] -> [B, 2048]
        v_concat = torch.cat([v1, v2, v3, v4], dim=1)
        
        # Final projection and L2 Normalization
        f = self.bn(self.proj(v_concat))
        return F.normalize(f, p=2.0, dim=1)


class CrossViewModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # Independent weights per branch, but UAV gets the Instance Norm injection
        self.uav_enc = ResNetFPNEncoder(embed_dim=embed_dim, is_uav=True)
        self.sat_enc = ResNetFPNEncoder(embed_dim=embed_dim, is_uav=False)
        
    def forward(self, uav, sat):
        return self.uav_enc(uav), self.sat_enc(sat)
    
    def encode_uav(self, x): return self.uav_enc(x)
    def encode_sat(self, x): return self.sat_enc(x)