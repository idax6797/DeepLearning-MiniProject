#!/bin/bash

echo "🧹 CLEAN INSTALLATION - Removing all packages first..."
echo "============================================================"

# Uninstall all relevant packages
echo "📦 Uninstalling existing packages..."
pip uninstall -y torch torchvision torchaudio numpy pandas scikit-learn scipy matplotlib seaborn Pillow opencv-python tqdm jupyter ipykernel

# Clear pip cache
echo "🗑️  Clearing pip cache..."
pip cache purge

# Update pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

echo ""
echo "✅ Clean slate ready! Now installing fresh packages..."
echo "============================================================"

# Install PyTorch FIRST
echo "📦 Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install NumPy
echo "📦 Installing NumPy (latest compatible)..."
pip install "numpy>=1.24.0,<2.0" --no-cache-dir

# Install scientific packages (compile against current NumPy)
echo "📦 Installing scientific packages..."
pip install --no-binary pandas,scikit-learn pandas scikit-learn scipy --no-cache-dir

# Install visualization & utilities
echo "📦 Installing visualization packages..."
pip install matplotlib seaborn Pillow opencv-python tqdm

# Install Jupyter
echo "📦 Installing Jupyter..."
pip install jupyter ipykernel

# Verify installations
echo ""
echo "============================================================"
echo "✅ Verifying installations..."
echo "============================================================"
python3 << 'EOF'
import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ NumPy: {np.__version__}")
print(f"✓ pandas: {pd.__version__}")
print(f"✓ scikit-learn: {sklearn.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

print("\n🧪 Testing imports...")
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
print("✓ All imports successful!")
EOF

echo ""
echo "🎉 Setup complete!"