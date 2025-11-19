# PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Spatio-Temporal State Space Model

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![arXiv](https://img.shields.io/badge/arXiv-2408.03540-b31b1b.svg)](https://arxiv.org/abs/2408.03540)

This is the official PyTorch implementation of our AAAI 2025 paper *"[PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Spatio-Temporal State Space Model](https://arxiv.org/pdf/2408.03540v2)"*.

**Forked and enhanced for DGX Spark with Docker support, CUDA kernel compilation fixes, and MLflow integration.**

## 🚀 Quick Start with Docker (DGX Spark)

This repository has been tested and optimized for **NVIDIA DGX Spark** systems with **NVIDIA GB10 (Blackwell)** GPUs.

### Prerequisites

- **Hardware:** NVIDIA DGX Spark with CUDA-capable GPU
- **Docker:** With NVIDIA Container Toolkit installed
- **CUDA:** 13.0+ (driver 580.95.05+)

### Docker Setup

**1. Start Docker Container with GPU Access:**

```bash
docker run -it --gpus all \
  --shm-size=8g \
  --security-opt label=disable \
  -v /home/tcong:/workspace \
  --name posemamba \
  nvcr.io/nvidia/pytorch:25.09-py3 \
  /bin/bash
```

**Key Settings:**
- `--gpus all`: Enables GPU access
- `--shm-size=8g`: Required for DataLoader workers
- `-v /home/tcong:/workspace`: Mounts your project directory
- Base image: `nvcr.io/nvidia/pytorch:25.09-py3` (PyTorch 2.9.0, CUDA 13.0.1, Python 3.12.3)

**2. Inside Container - Install Dependencies:**

```bash
cd /workspace/Documents/realposemamba/PoseMamba

# Install Python dependencies
pip install -r requirements.txt
pip install mlflow tensorboardX

# Compile CUDA kernels (CRITICAL for performance!)
cd kernels/selective_scan
pip install -e .
cd ../..
```

**3. Verify GPU Access:**

```bash
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GB10
```

**4. Verify CUDA Kernels Compiled:**

```bash
python3 -c "from lib.model.mambablocks import selective_scan_cuda_oflex_available; print(f'CUDA kernels: {selective_scan_cuda_oflex_available}')"
```

Expected: `CUDA kernels: True` ✅

---

## 📦 Environment

### Docker Environment (Recommended for DGX Spark)

- **Base Image:** `nvcr.io/nvidia/pytorch:25.09-py3`
- **Python:** 3.12.3
- **PyTorch:** 2.9.0a0+50eac811a6.nv25.9
- **CUDA:** 13.0.1
- **OS:** Ubuntu 24.04.3 LTS

### Local Environment (Original)

The project was originally developed under:
- Python 3.8.5
- PyTorch 1.13.1+cu117
- CUDA 11.7

For local installation (without Docker):

```bash
conda create -n posemamba python=3.8.5
conda activate posemamba
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
cd kernels/selective_scan && pip install -e .
```

---

## 📊 Dataset

### Human3.6M

#### Preprocessing

1. Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to `data/motion3d`, or direct download our processed data [here](https://drive.google.com/file/d/1WWoVAae7YKKKZpa1goO_7YcwVFNR528S/view?usp=sharing) and unzip it.

2. Slice the motion clips:

```bash
# Inside Docker container
python tools/convert_h36m.py
```

### MPI-INF-3DHP

Please refer to [MotionAGFormer](https://github.com/taatiteam/motionagformer) for dataset setup.

---

## 🏋️ Training

### Quick Test (1 Epoch)

**Inside Docker container:**

```bash
cd /workspace/Documents/realposemamba/PoseMamba
PYTHONPATH=. python train.py \
  --config configs/pose3d/PoseMamba_test_h36m.yaml \
  --checkpoint checkpoint_test
```

**Expected time:** 20-30 minutes (with compiled CUDA kernels)

### Full Training

**Human3.6M:**

```bash
# Inside Docker container
PYTHONPATH=. python train.py \
  --config configs/pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint checkpoint_full_training
```

**Training Configuration:**
- **Epochs:** 120
- **Batch Size:** 4
- **Learning Rate:** 0.0002
- **Sequence Length:** 243 frames
- **Expected Duration:** ~1.9 days (120 epochs)

**Monitor Training:**

```bash
# In another terminal (on host)
cd /home/tcong/Documents/realposemamba/PoseMamba
mlflow ui --backend-store-uri ./mlruns
# Open: http://localhost:5000
```

### MLflow Integration

This fork includes **MLflow tracking** for experiment management:

- **Automatic logging:** Parameters, metrics, checkpoints, configs
- **Real-time monitoring:** View training progress in MLflow UI
- **Experiment tracking:** Compare different training runs

MLflow data is stored in `./mlruns/` directory.

---

## 🔧 Key Fixes and Enhancements

This fork includes several critical fixes for DGX Spark:

### 1. CUDA Kernel Compilation Fixes

**Files Modified:**
- `kernels/selective_scan/setup.py`: Added `-Wno-narrowing` flag
- `kernels/selective_scan/csrc/selective_scan/cus/selective_scan.cpp`: Fixed narrowing conversion
- `kernels/selective_scan/csrc/selective_scan/cusoflex/selective_scan_oflex.cpp`: Fixed narrowing conversion
- `kernels/selective_scan/csrc/selective_scan/reverse_scan.cuh`: Fixed CUB API compatibility (CUDA 13.0)

**Impact:** Enables successful compilation of CUDA kernels, providing 10-30x speedup.

### 2. Model Architecture Fix

**File Modified:** `lib/model/PoseMamba.py`

Changed `forward_type` from `'v2_plus_poselimbs'` to `'v3'` to use compiled CUDA kernels (`SelectiveScanOflex`) instead of CPU fallback.

**Impact:** Enables GPU acceleration for selective scan operations.

### 3. MLflow Integration

**File Modified:** `train.py`

Added comprehensive MLflow tracking:
- Parameter logging (all hyperparameters)
- Metric logging (losses, MPJPE, P-MPJPE)
- Artifact logging (checkpoints, logs, configs)
- Experiment management

**Impact:** Full experiment tracking and reproducibility.

### 4. PyTorch 2.6 Compatibility

**File Modified:** `train.py`

Added `weights_only=False` to all `torch.load()` calls for PyTorch 2.6+ compatibility.

### 5. Additional Model Fixes

**Files Modified:**
- `lib/model/csms6s.py`: Enhanced selective scan implementations
- `lib/model/mambablocks.py`: Improved BiSTSSM block configuration

---

## 📈 Performance

### Training Speed (DGX Spark, NVIDIA GB10)

**With Compiled CUDA Kernels:**
- **Iteration Time:** ~0.5-1.0 seconds per batch
- **GPU Utilization:** ~92%
- **Throughput:** ~4-8 samples/second
- **Epoch Time:** ~20-25 minutes per epoch

**Memory Usage:**
- **GPU Memory:** ~2.7 GB (model + active batch)
- **System RAM:** ~117 GB (dataset + DataLoader workers + cache)

### Training Results (PoseMamba-S, 120 epochs)

- **Final MPJPE:** 43.63mm
- **Best MPJPE:** 41.96mm
- **P-MPJPE:** 35.47mm
- **Training Duration:** 1.9 days

---

## 🧪 Evaluation

We provide [checkpoint](https://drive.google.com/file/d/1WFRAeal8W6ntrTPNrf-SNywdgupj0-S8/view?usp=sharing). You can download and unzip it to get pretrained weight.

| Method      | frames | Params | MACs  | Human3.6M weights                                                                                     |
| ----------- | ------ | ------ | ----- | ----------------------------------------------------------------------------------------------------- |
| PoseMamba-S | 243    | 0.9M   | 3.6G  | [PoseMamba-S](https://drive.google.com/file/d/1LZtEjeiAIx6LXFmjoyKKzbaCPV3R1-P7/view?usp=sharing)     |
| PoseMamba-B | 243    | 3.4M   | 13.9G | [PoseMamba-B](https://drive.google.com/file/d/1aP6WAq5fKNIqyYcI%5FZnYbuagR3%5FzVik2/view?usp=sharing) |
| PoseMamba-L | 243    | 6.7M   | 27.9G | [PoseMamba-L](https://drive.google.com/file/d/16%5FTg0Aqzgih243%5DdflyFv0UB79gU9u8q/view?usp=sharing) |

After downloading the weight, you can evaluate Human3.6M models:

```bash
# Inside Docker container
python train.py --eval-only \
  --checkpoint <CHECKPOINT-DIRECTORY> \
  --checkpoint-file <CHECKPOINT-FILE-NAME> \
  --config <PATH-TO-CONFIG>
```

For example:

```bash
python train.py --eval-only \
  --checkpoint checkpoint \
  --checkpoint-file PoseMamba-l-h36m.pth.tr \
  --config configs/pose3d/PoseMamba_train_h36m_S.yaml
```

---

## 🎬 Demo

Our demo is a modified version of the one provided by [MotionAGFormer](https://github.com/taatiteam/motionagformer) repository. 

**Setup:**
1. Download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in `./demo/lib/checkpoint`
2. Download our base model checkpoint from [here](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view) and put it in `./checkpoint`
3. Put your in-the-wild videos in `./demo/video`
4. Download [demo files](https://drive.google.com/file/d/1hbK1HDz1nMTGYcczOC5r33Mk8nAtLZCr/view?usp=sharing) and unzip

**Run:**

```bash
# Inside Docker container
python vis.py --video sample_video.mp4 --gpu 0
```

Sample demo output:

<p align="center">
<img src='sample_video.gif' width="60%" alt="no img" />
</p>

---

## 📚 Additional Documentation

This fork includes extensive documentation:

- **`PROJECT_COMPLETE_SUMMARY.md`**: Complete project overview with architecture diagrams
- **`DOCKER_QUICK_START.md`**: Detailed Docker setup guide
- **`MASTER_GUIDE.md`**: High-level workflow guide
- **`SPEED_FIX_FINAL.md`**: CUDA kernel compilation guide
- **`MLFLOW_INTEGRATION_GUIDE.md`**: MLflow usage guide

---

## 🐛 Troubleshooting

### CUDA Kernels Not Compiled

**Symptom:** Training is very slow (~3 sec/batch instead of ~0.5 sec/batch)

**Solution:**
```bash
# Inside Docker container
cd kernels/selective_scan
pip install -e .
```

### GPU Not Detected

**Check:**
```bash
docker exec posemamba nvidia-smi
```

**If no GPU:** Ensure Docker was started with `--gpus all` flag.

### MLflow UI Not Accessible

**Solution:**
```bash
# On host machine (not in Docker)
cd /home/tcong/Documents/realposemamba/PoseMamba
mlflow ui --backend-store-uri ./mlruns
```

---

## 🙏 Acknowledgement

Our code refers to the following repositories:

- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [MotionAGFormer](https://github.com/taatiteam/MotionAGFormer)
- [VMamba](https://github.com/mzeromiko/vmamba)

We thank the authors for releasing their codes.

---

## 📝 Citation

If you find our work useful for your project, please consider citing the paper:

```bibtex
@article{huang2024posemamba,
  title={PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model},
  author={Huang, Yunlong and Liu, Junshuo and Xian, Ke and Qiu, Robert Caiming},
  journal={arXiv preprint arXiv:2408.03540},
  year={2024}
}
```

---

## 🔗 Fork Information

**Original Repository:** [nankingjing/PoseMamba](https://github.com/nankingjing/PoseMamba)

**This Fork:** [trevcong/PA-PoseMamba](https://github.com/trevcong/PA-PoseMamba)

**Enhancements:**
- ✅ Docker support for DGX Spark
- ✅ CUDA 13.0 compatibility fixes
- ✅ MLflow integration
- ✅ PyTorch 2.6+ compatibility
- ✅ Performance optimizations
- ✅ Comprehensive documentation

---

## 📧 Contact

For issues specific to this fork, please open an issue on [GitHub](https://github.com/trevcong/PA-PoseMamba/issues).
