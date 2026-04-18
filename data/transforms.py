import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class ApplyCLAHE:
    """Normalizes exposure variation across UAV images taken at different lighting conditions."""
    def __call__(self, img):
        # Convert PIL Image (RGB) to OpenCV format (BGR)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE in LAB color space (as requested in your guidelines)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab_clahe = cv2.merge([clahe.apply(l), a, b])
        bgr_out = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        # Convert back to PIL Image (RGB)
        rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_out)

class ApplyPolarTransform:
    """Simulates nadir satellite viewpoint from oblique UAV perspectives."""
    def __call__(self, img):
        cv_img = np.array(img)
        h, w = cv_img.shape[:2]
        center = (w / 2, h / 2)
        max_radius = np.sqrt(((w / 2.0) ** 2.0) + ((h / 2.0) ** 2.0))
        polar_img = cv2.warpPolar(cv_img, (w, h), center, max_radius, cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
        return Image.fromarray(polar_img)

def uav_transform(img_size=224):
    return T.Compose([
        ApplyCLAHE(), # Applied first to normalize lighting
        T.RandomApply([ApplyPolarTransform()], p=0.5), # 50% chance for Polar Warp
        T.RandomRotation(180),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomPerspective(distortion_scale=0.2),
        T.RandomAffine(degrees=0, scale=(0.7, 1.3)), # Altitude Scale Jitter
        T.ColorJitter(0.2, 0.2, 0.1, 0.05),
        T.GaussianBlur(3, sigma=(0.1, 2.0)),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

def sat_transform(img_size=224):
    return T.Compose([
        T.RandomChoice([
            T.Lambda(lambda x: x),
            T.RandomRotation((90, 90)), 
            T.RandomRotation((180, 180)), 
            T.RandomRotation((270, 270))
        ]),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1, 0.1, 0.0, 0.0),
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])