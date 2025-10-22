#!/bin/bash

# UCloud Setup Script for U-Net Tumor Segmentation
echo "🚀 Setting up environment for U-Net Training..."
echo "============================================================"

# Update pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# CRITICAL: Install NumPy first to avoid compatibility issues
echo "📦 Installing NumPy (compatible version)..."
pip install "numpy>=1.23.0,<2.0"

# Install PyTorch with CUDA support (for GPU acceleration on UCloud)
echo "📦 Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch

# Install data processing and visualization (after NumPy)
echo "📦 Installing data processing packages..."
pip install "pandas>=1.5.0,<2.0"
pip install "matplotlib>=3.5.0"
pip install seaborn
pip install Pillow
pip install opencv-python

# OpenCV is optional - only install if needed for your specific use case
# pip install opencv-python-headless

# Install ML/DL utilities
echo "📦 Installing scikit-learn..."
pip install scikit-learn
pip install seaborn

# Install progress bar
echo "📦 Installing tqdm..."
pip install tqdm

# Install Jupyter
echo "📦 Installing Jupyter..."
pip install jupyter ipykernel

# Verify installations
echo ""
echo "✅ Verifying installations..."
python3 << EOF
import torch
import numpy as np
import matplotlib
import tqdm
import sklearn

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU devices: {torch.cuda.device_count()}")
    print(f"✓ Current GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ NumPy version: {np.__version__}")
print(f"✓ Matplotlib version: {matplotlib.__version__}")
print(f"✓ scikit-learn version: {sklearn.__version__}")
EOF

echo ""
echo "🎉 Setup complete! Ready to run U-Net training."
echo "📊 Your data structure should be:"
echo "   augmented_data/"
echo "   ├── patients/"
echo "   │   ├── imgs/"
echo "   │   └── labels/"
echo "   └── controls/"
echo "       └── imgs/"