# 🧠 Brain Tumor Classification - Computer Vision & Deep Learning

A medical imaging project that compares classical computer vision techniques with modern deep learning for automated brain tumor detection from MRI scans.

[![Python](https://img.shields.io/badge/Python-3.11-atestcy-96.23%25-
***

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

***

## 🎯 Overview

This project tackles brain tumor detection using two different approaches:

**Classical Approach**: Traditional computer vision feature extraction (HOG, GLCM, LBP) combined with machine learning classifiers (Random Forest, XGBoost). This method achieved approximately **52% accuracy**.

**Deep Learning Approach**: Transfer learning using pre-trained VGGNet-16 architecture, fine-tuned on brain MRI data. This method achieved **96.23% accuracy**.

The project demonstrates how deep learning significantly outperforms traditional methods for medical image analysis.

***

## 📁 Project Structure

```
BrainTumor-Classification---Computer-Vision/
│
├── ML/
│   ├── Initial_Setup.ipynb              # Data loading and preprocessing
│   └── Final_ML_Classification.ipynb    # Classical ML approach
│
├── brain-tumor/                         # Dataset folder
│   ├── images/
│   │   ├── train/                       # 893 training MRI images
│   │   └── val/                         # 267 validation MRI images
│   └── labels/
│       ├── train/                       # Training labels (YOLO format)
│       └── val/                         # Validation labels (YOLO format)
│
├── final-vggnet.ipynb                   # VGGNet-16 implementation (Best model)
├── googlenet.ipynb                      # GoogLeNet implementation
└── README.md
```

***

## 📊 Dataset

**Task**: Binary classification (Tumor vs No Tumor)

**Original Dataset**:
- Training: 893 MRI scans
- Validation: 267 MRI scans
- Format: Grayscale images with YOLO bounding box annotations

**After Preprocessing** (patch-based approach):
- Training patches: 31,506 (No Tumor: 28,680, Tumor: 2,826)
- Validation patches: 8,407 (No Tumor: 7,743, Tumor: 664)
- Patch size: 64×64 pixels, resized to 224×224 for CNN input

**Note**: There is a significant class imbalance with approximately 10 times more non-tumor patches than tumor patches.

***

## 🔬 Methodology

### **Phase 1: Classical Computer Vision**

#### **Preprocessing**
1. Convert images to grayscale
2. Extract 64×64 patches using sliding window (stride=64)
3. Check overlap with tumor bounding boxes for labeling
4. Resize patches to 224×224
5. Normalize pixel values

#### **Feature Extraction**
- **HOG (Histogram of Oriented Gradients)**: Captures edge and shape information
- **GLCM (Gray-Level Co-occurrence Matrix)**: Extracts texture features like contrast and energy
- **LBP (Local Binary Patterns)**: Provides rotation-invariant texture patterns
- **Statistical features**: Mean, standard deviation, skewness, kurtosis
- **Histogram features**: 16-bin intensity distributions

#### **Classifiers**
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting framework

**Result**: Both achieved approximately **52% validation accuracy**

***

### **Phase 2: Deep Learning**

#### **VGGNet-16 (Best Model)**

**Architecture**:
- Pre-trained VGG-16 from ImageNet (trained on 1.2M images)
- 13 convolutional layers + 3 fully connected layers
- Modified final layer: 1000 classes → 2 classes (binary)

**Training Setup**:
- Optimizer: Adam (learning rate = 0.0001)
- Loss: Cross-Entropy
- Batch size: 32
- Epochs: 5
- Device: GPU (CUDA)

**Data Processing**:
- Extract patches from MRI scans
- Convert grayscale to RGB (replicate across 3 channels)
- Normalize with mean=(0.5, 0.5, 0.5)

**Why Transfer Learning Works**:
- Pre-trained layers already detect edges, textures, and patterns
- Only final classification layer needs retraining for tumor detection
- Dramatically reduces training time and improves accuracy

**Result**: **96.23% validation accuracy**

#### **GoogLeNet**
Alternative architecture using Inception modules for multi-scale feature extraction (see googlenet.ipynb for details)

***

## 🚀 Installation

### **Requirements**
- Python 3.11+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### **Step 1: Clone Repository**
```bash
git clone https://github.com/Seventie/BrainTumor-Classification---Computer-Vision.git
cd BrainTumor-Classification---Computer-Vision
```

### **Step 2: Create Environment**
```bash
# Using conda (recommended)
conda create -n brain-tumor python=3.11
conda activate brain-tumor

# OR using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### **Step 3: Install Dependencies**

**PyTorch (with CUDA support)**:
```bash
# Check your CUDA version first
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# OR using pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Other Libraries**:
```bash
pip install opencv-python scikit-learn scikit-image xgboost numpy scipy pandas matplotlib seaborn jupyter notebook
```

**Optional GPU Acceleration**:
```bash
pip install cuml-cu11 cupy-cuda11x
```

### **Step 4: Prepare Dataset**

Place your dataset in this structure:
```
brain-tumor/
├── images/
│   ├── train/  ← Place training .jpg/.png files here
│   └── val/    ← Place validation .jpg/.png files here
└── labels/
    ├── train/  ← Place corresponding .txt YOLO files here
    └── val/    ← Place corresponding .txt YOLO files here
```

**YOLO Label Format** (each line in .txt file):
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized between 0-1. Example:
```
1 0.512 0.487 0.234 0.198
```

***

## 💻 Usage

### **Launch Jupyter Notebooks**
```bash
jupyter notebook
```

### **Recommended Workflow**

**1. Data Setup** → Open `Initial_Setup.ipynb`
- Load dataset
- Verify file paths
- Explore data distribution
- Visualize sample images with annotations

**2. Classical ML** → Open `Final_ML_Classification.ipynb`
- Extract features (HOG, GLCM, LBP)
- Train Random Forest and XGBoost
- Evaluate performance (~52% accuracy)
- Analyze feature importance

**3. Deep Learning** → Open `final-vggnet.ipynb`
- Load pre-trained VGG-16
- Fine-tune on brain tumor patches
- Monitor training progress
- Evaluate results (96.23% accuracy)
- Visualize predictions

**4. Alternative Model** → Open `googlenet.ipynb`
- Explore GoogLeNet architecture
- Compare with VGGNet results

***

## 📈 Results

### **Performance Comparison**


✅ Deep learning outperforms classical methods by **30%**  
✅ Transfer learning accelerates training dramatically  
✅ Minimal overfitting (only 1.5% gap between train/val)  
✅ Handles class imbalance effectively  
✅ Suitable for clinical decision support applications

***

## 🔧 How It Works

### **Classical ML Pipeline**
1. Extract patches from MRI scans
2. Compute hand-crafted features (HOG, GLCM, LBP)
3. Train traditional classifiers
4. Predict tumor presence

### **Deep Learning Pipeline**
1. Extract patches from MRI scans
2. Convert to 3-channel RGB format
3. Feed through pre-trained VGG-16
4. Fine-tune final layers
5. Classify as tumor/non-tumor

### **Why Deep Learning Wins**
- Automatically learns hierarchical features
- Pre-trained on millions of images
- Captures complex patterns humans can't define
- Adapts to medical imaging domain through fine-tuning

***

## 🔮 Future Improvements

- Add data augmentation (rotation, flipping, brightness)
- Address class imbalance with weighted loss
- Test other architectures (ResNet, EfficientNet)
- Implement multi-class classification
- Add explainability (Grad-CAM visualizations)
- Deploy as web application

***

## 📄 License

MIT License - Free to use and modify

***

**Repository**: [github.com/Seventie/BrainTumor-Classification---Computer-Vision](https://github.com/Seventie/BrainTumor-Classification---Computer-Vision)

⭐ **Star this repo if you found it helpful!**
