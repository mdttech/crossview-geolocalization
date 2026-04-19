# 🚁 Cross-View Geo-Localization: UAV-to-Satellite Retrieval

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2FGPU-ee4c2c)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff69b4)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-5c629e)

**A Dual-Branch CNN Framework with Edge-Optimized Deployment for UAV-to-Satellite Retrieval in GPS-Denied Environments.**

## 📖 Overview
Visual geo-localization enables autonomous Unmanned Aerial Vehicle (UAV) navigation in environments where GPS signals are jammed, spoofed, or unavailable. This project solves the geometric challenge of matching real-time, low-altitude, 45-degree angled drone imagery against a massive gallery of 90-degree top-down satellite tiles.

While recent trends favor Vision Transformers (ViTs) and semantic models like CLIP, this project demonstrates that a meticulously optimized Convolutional Neural Network (CNN) remains mathematically superior for strict spatial and geometric matching.  
<br>

<img width="1455" height="161" alt="Untitled Diagram drawio" src="https://github.com/user-attachments/assets/a9953517-6b99-4593-af2e-73e6729305e5" />

<br>

Furthermore, this repository bridges the gap between academic theory and edge hardware, featuring a fully quantized INT8 deployment pipeline and a sub-2ms FAISS vector database.

<br>

<img width="680" height="79" alt="Untitled Diagram drawio (1)" src="https://github.com/user-attachments/assets/7b1a4e49-13f6-42eb-8f2c-a139875b9f28" />

<br>

---

## ✨ Key Features & Architectural Novelties
1. **Style-Agnostic Learning:** Injected **Instance Normalization** into the UAV feature extractor to mathematically strip dynamic environmental lighting/weather variances, forcing the network to align pure structural geometry.
2. **Multi-Scale Fusion:** Integrated a **Feature Pyramid Network (FPN)** to seamlessly bridge the massive scale discrepancy between drone and satellite imagery.
3. **Optimized Pooling:** Utilized **Generalized Mean (GeM) Pooling ($p=3.0$)** combined with a symmetric **InfoNCE Contrastive Loss** to map completely different viewpoints into a shared 512D spatial dimension.
4. **Edge-Compute Ready:** Compiled the heavy PyTorch architecture into a highly compressed **INT8 TorchScript** model, reducing the memory footprint by 6x for deployment on companion computers (e.g., Raspberry Pi, Jetson Nano).

---

## 📊 Experimental Results & Ablation Study

Evaluated on the **University-1652 Dataset** (701 unique buildings), our architecture was subjected to a rigorous, seed-locked ablation tournament against state-of-the-art architectures.

### The Backbone Showdown (512-Dim)
| Backbone Architecture | Loss Function | Recall@1 |
| :--- | :--- | :--- |
| **ResNet50 + FPN (Ours)** | **InfoNCE** | **89.07%** |
| ViT-Small (Patch16) | InfoNCE | 83.73% |
| ViT-Base (Patch16) | InfoNCE | 80.50% |

*Note: Peak Validation Recall@1 reached **89.81%** at Epoch 39 during extended fine-tuning.*

<img width="705" height="461" alt="training_curve" src="https://github.com/user-attachments/assets/91e6ab21-fdd9-4464-b098-70dbd489f620" />


### Edge Deployment Performance
To simulate real-world drone deployment, the UAV encoder was quantized to INT8 and matched against 701 pre-computed satellite embeddings in a FAISS (Flat L2) index.
* **Memory Footprint:** 608.1 MB $\rightarrow$ **101.6 MB** (6x Reduction)
* **Inference + Retrieval Latency:** **1.76 ms** (CPU only)
* **Zero-Shot Deployment Recall@5:** **86.98%** (Ensuring the correct GPS coordinate is almost always within the top 5 instantaneous returns).

---

## 🛠️ Repository Structure

```text
crossview-geolocalization/
│
├── configs/
│   ├── baseline.yaml               # Hyperparameters for the core ResNet-FPN model
│   └── vit_infonce.yaml            # Hyperparameters for the Vision Transformer baseline
│
├── data/
│   ├── splits/
│   │   ├── train_locs.npy          # Numpy array of location IDs used for training
│   │   ├── val_locs.npy            # Numpy array of location IDs used for validation
│   │   └── test_locs.npy           # Numpy array of location IDs used for final testing
│   │
│   ├── crossview_dataset.py        # PyTorch Dataset class for loading and pairing UAV/Satellite imagery
│   ├── gps_labels.csv              # Ground-truth mapping of location IDs to Latitude/Longitude coordinates
│   └── transforms.py               # Image preprocessing, resizing, and augmentation pipelines
│
├── evaluation/
│   └── metrics.py                  # Functions to calculate Recall@K and mean Average Precision (mAP)
│
├── models/
│   ├── crossview_clip.py           # OpenAI CLIP architecture implementation for baseline comparison
│   ├── crossview_model.py          # The core dual-branch ResNet50 + FPN architecture with Instance Normalization
│   ├── crossview_vit.py            # Vision Transformer architecture for ablation studies
│   └── losses.py                   # Custom loss functions, including the symmetric InfoNCE Contrastive Loss
│
├── notebooks/
│   └── check_dataset.ipynb         # Jupyter notebook for visually inspecting data pairs and augmentations
│
├── results/
│   ├── ablation_results.csv        # Final compiled metrics comparing backbones, losses, and embedding dimensions
│   ├── training_log_resnet50.csv   # Epoch-by-epoch training and validation metrics for the ResNet model
│   └── training_log_vit.csv        # Epoch-by-epoch training logs for the ViT baseline
│
├── scripts/
│   ├── build_deployment.py         # Compiles the INT8 quantized model and pre-computes the FAISS vector index
│   ├── final_eval.py               # Evaluates the edge-deployed model for latency, Recall@K, and mAP metrics
│   ├── generate_splits.py          # Utility to divide the dataset into train, validation, and test numpy arrays
│   ├── preprocess_dataset.py       # Handles initial data cleaning, resizing, and formatting of the raw aerial imagery
│   ├── run_ablation.py             # Automated pipeline for the seed-locked ablation tournament across architectures
│   ├── test_dataloader.py          # Debugging script to verify image pairing, batch generation, and augmentations
│   ├── train.py                    # Main training loop and optimization logic for the core ResNet50 + FPN model
│   ├── train_mae.py                # Experimental script for Masked Autoencoder (MAE) pre-training pipelines
│   ├── train_vit.py                # Dedicated training loop for the Vision Transformer (ViT) baseline models
│   └── verify_clip.py              # Evaluates the zero-shot geometric spatial reasoning of the OpenAI CLIP baseline
│
├── check_data.py                   # Utility script to verify dataset integrity, image counts, and folder structures
├── make_gps_csv.py                 # Script to extract and compile ground-truth GPS coordinates into the master CSV
├── requirements.txt                # List of all Python dependencies needed to run the project
└── .gitignore                      # Instructs Git to ignore heavy dataset folders and large .pth model weights
│
├── requirements.txt                # List of all Python dependencies needed to run the project
└── .gitignore                      # Instructs Git to ignore heavy dataset folders and large .pth model weights
```
> **Note:** To comply with GitHub file size limits, large dataset folders and `.pth` model weight files are excluded from version control using `.gitignore`.

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/mdttech/crossview-geolocalization.git](https://github.com/mdttech/crossview-geolocalization.git)
cd crossview-geolocalization
```

### 2. Set Up the Environment (Miniconda Recommended)
```bash
conda create -n crossview python=3.10 -y
conda activate crossview
```

### 3. Install Dependencies
Install the core deep learning and vector search packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

(Note: If running or testing the deployment pipeline locally on Windows, ensure you manually install faiss-cpu to avoid build errors).

---

## 🏃‍♂️ Usage & Pipeline Execution

This repository is built as an end-to-end Machine Learning pipeline. Once your environment is set up and the University-1652 dataset is placed in the `data/datasets/` directory, you can execute the following phases:

---

### Phase 1: Data Preparation

Run these scripts from the root directory to clean, format, and verify the dataset before training.

```bash
# 1. Preprocess and clean the raw aerial imagery
python scripts/preprocess_dataset.py

# 2. Extract and compile ground-truth GPS coordinates
python make_gps_csv.py

# 3. Generate train/val/test numpy dataset splits
python scripts/generate_splits.py

# 4. Verify dataset integrity and image pairing
python check_data.py
```

---

### Phase 2: Training & Ablation

You can train individual models using the provided YAML configuration files, or run the full ablation tournament.

```bash
# Train the core ResNet50 + FPN model
python scripts/train.py --config configs/baseline.yaml

# Run the automated ablation tournament (CNN vs. ViT)
python scripts/run_ablation.py
```

---

### Phase 3: Edge Deployment Simulation

To compress the trained model and test it against edge-hardware constraints:

```bash
# Quantize the UAV encoder to INT8 and build the 701-building FAISS index
python scripts/build_deployment.py

# Evaluate latency and Recall@K metrics on the fully compressed pipeline
python scripts/final_eval.py
```

---

## 📚 Datasets & Acknowledgments

University-1652 Dataset: Used for all training, validation, and geo-localization testing. (Zheng et al., ACM Multimedia 2020)

FAISS: Facebook AI Similarity Search used for sub-millisecond similarity search indexing.

---

**Author:** Tahseen  
**License:** MIT





