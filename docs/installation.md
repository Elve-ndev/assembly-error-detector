# Installation Guide

## System Requirements

### Hardware

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU | NVIDIA RTX 2060 (6GB) | RTX 3090 (24GB) | CUDA Compute 6.1+ |
| RAM | 16GB | 32GB | For data loading & processing |
| Storage | 20GB | 50GB | Model weights + datasets |
| CPU | Intel i7 / AMD Ryzen 5 | i9 / Ryzen 9 | For preprocessing |

### Software

- **Python**: 3.10+ (tested on 3.10, 3.11, 3.13)
- **CUDA**: 11.8+ (for GPU acceleration)
- **cuDNN**: 8.6+ (for efficient GPU operations)
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows (with WSL2)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Elve-ndev/assembly-error-detector.git
cd assembly-error-detector
```

### 2. Create Virtual Environment

```bash
# Using venv
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n assembly-detector python=3.11
conda activate assembly-detector
```

### 3. Install PyTorch with CUDA Support

```bash
# For CUDA 11.8 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (slower, for testing)
pip install torch torchvision torchaudio
```

### 4. Install Project Dependencies

```bash
pip install -e .
```

Or manually install requirements:

```bash
pip install numpy scipy scikit-learn pandas matplotlib opencv-python
pip install PyYAML tqdm tensorboard
pip install optuna wandb  # For hyperparameter tuning & logging
```

### 5. Setup CUDA (GPU Users)

**Verify CUDA Installation:**

```bash
# Check NVIDIA GPU
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.104.05    Driver Version: 535.104.05    CUDA Version: 12.2   |
# +-----------------------------------------------------------------------------+
# | GPU  Name            TCC.SM  Memory-Usage | GPU-Util  Compute M. |
# +=============================================================================+
# |   0  NVIDIA RTX 3090     ... |  2200MiB / 24576MiB |  45%   Default |
# +-----------------------------------------------------------------------------+
```

**Set CUDA Environment Variables:**

```bash
# Linux/macOS
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version  # Should show CUDA Toolkit version
```

### 6. Download Pre-trained Models

**Option A: Download from Release**

```bash
wget https://github.com/Elve-ndev/assembly-error-detector/releases/download/v1.0/slowfast_r50_meccano.pth -O models/slowfast_r50_meccano.pth
wget https://github.com/Elve-ndev/assembly-error-detector/releases/download/v1.0/bigru_dualhead.pth -O models/bigru_dualhead.pth
```

**Option B: Download Manually**

1. Visit [Releases](https://github.com/Elve-ndev/assembly-error-detector/releases)
2. Download model files to `models/` directory

### 7. Download Datasets

**IndustReal Dataset:**

```bash
# Clone the IndustReal repository
git clone https://github.com/TimSchoonbeek/IndustReal.git
cd IndustReal

# Follow their setup instructions
# Then link to your assembly-error-detector project
ln -s /path/to/IndustReal ../assembly-error-detector/data/industreal
```

**Or Download Directly:**

```bash
# Create data directory
mkdir -p data/industreal

# Download from official source (follow IndustReal GitHub)
# Place videos in: data/industreal/videos/
# Place annotations in: data/industreal/annotations/
```

### 8. Verify Installation

```bash
# Test imports
python -c "from assembly_error_detector import SlowFastFeatureExtractor; print('✅ Installation successful!')"

# Run basic test
python -m pytest tests/ -v

# GPU availability check
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Configuration

### Create Configuration File

Create `config.yaml` in project root:

```yaml
# config.yaml
model:
  slowfast_weights: "models/slowfast_r50_meccano.pth"
  bigru_hidden_dim: 256
  bigru_num_layers: 2
  dropout_rate: 0.3
  
feature_extraction:
  frame_size: 224
  num_frames: 8
  fps: 30
  normalize: true
  
training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 0.001
  weight_decay: 1e-5
  device: "cuda"  # "cpu" for CPU-only
  
anomaly:
  method: "mahalanobis"  # or "isolation_forest"
  threshold: 0.65
  
cobot:
  critical_threshold: 0.85
  pause_threshold: 0.65
  monitor_threshold: 0.45
```

### Environment Variables

```bash
# Optional: Set default device
export ASSEMBLY_DEVICE="cuda"

# Optional: Set data path
export ASSEMBLY_DATA_PATH="/path/to/data"

# Optional: Disable GPU (for testing)
export CUDA_VISIBLE_DEVICES=""
```

## Docker Installation (Optional)

**Build Docker Image:**

```bash
docker build -t assembly-error-detector:latest -f Dockerfile .
```

**Run with Docker:**

```bash
# With GPU support
docker run --gpus all -it -v $(pwd)/data:/workspace/data \
  assembly-error-detector:latest

# CPU-only
docker run -it -v $(pwd)/data:/workspace/data \
  assembly-error-detector:latest
```

## Troubleshooting

### Issue: CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate XX.XX GiB
```

**Solutions:**

1. **Reduce batch size** in `config.yaml`:
   ```yaml
   training:
     batch_size: 8  # Reduce from 16
   ```

2. **Clear GPU cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Use CPU mode** (slower):
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   ```

### Issue: Model Weights Not Found

```
FileNotFoundError: [Errno 2] No such file or directory: 'models/slowfast_r50_meccano.pth'
```

**Solutions:**

1. Ensure `models/` directory exists:
   ```bash
   mkdir -p models/
   ```

2. Download weights from Releases (see Step 6)

3. Update path in `config.yaml`:
   ```yaml
   model:
     slowfast_weights: "/absolute/path/to/slowfast_r50_meccano.pth"
   ```

### Issue: IndustReal Dataset Not Found

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/industreal/videos'
```

**Solutions:**

1. Download IndustReal dataset (see Step 7)

2. Verify structure:
   ```bash
   ls data/industreal/
   # Should show: videos/, annotations/, README.md, ...
   ```

3. Update path in `config.yaml`:
   ```yaml
   dataset:
     root_path: "/absolute/path/to/industreal"
   ```

### Issue: PyTorch Version Conflict

```
ImportError: cannot import name '_new_empty_tensor' from torch._C
```

**Solution:** Reinstall PyTorch correctly:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: OpenCV Not Working

```
ImportError: libGL.so.1: cannot open shared object file
```

**Solution (Linux):**

```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx libsm6 libxext6

# CentOS/RHEL
sudo yum install mesa-libGL libXext
```

## Verification Checklist

Run this script to verify everything is working:

```bash
python scripts/verify_installation.py
```

This checks:
- ✅ Python version
- ✅ PyTorch installation
- ✅ CUDA availability
- ✅ Required dependencies
- ✅ Model weights
- ✅ Dataset availability
- ✅ GPU memory

## Development Setup

For contributors:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Generate coverage report
pytest --cov=assembly_error_detector tests/

# Format code
black assembly_error_detector/ tests/

# Lint code
pylint assembly_error_detector/
```

## Next Steps

1. **Quick Start**: See [Usage Guide](usage.md)
2. **Train Model**: See [Training Tutorial](usage.md#training)
3. **Run Inference**: See [Inference Example](usage.md#inference)
4. **API Reference**: See [API Documentation](api.rst)

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Elve-ndev/assembly-error-detector/issues)
- **Discussions**: [Ask questions](https://github.com/Elve-ndev/assembly-error-detector/discussions)
- **Email**: Contact project maintainers
