# Forest Inspection - UAV Semantic Segmentation

Dự án semantic segmentation cho ảnh UAV giám sát rừng, sử dụng [Forest Inspection Dataset (Sunny sequences)](https://zenodo.org/records/15511426).

## 📋 Tổng quan

- **Dataset**: 38 GB, ~13,128 cặp ảnh RGB + label, 11 classes
- **Task**: Pixel-level semantic segmentation
- **Models**: U-Net, U-Net++, DeepLabV3+, HRNet
- **Paper**: [arXiv:2403.06621](https://arxiv.org/abs/2403.06621)

## 🚀 Quick Start

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Download data (hoặc dùng Kaggle)
python scripts/download_zenodo.py --seq 1 2 3

# 3. Khám phá dataset
python scripts/explore_dataset.py --data data/forest_sunny --show-samples 3

# 4. Train
python train.py --config configs/train_unet.yaml

# 5. Train trên Kaggle (override data path)
python train.py --config configs/train_unet.yaml --data-root /kaggle/input/forest-sunny
```

## � Offline Kaggle Training (NEW!)

Để train model **OFFLINE** trên Kaggle với RTX 6900 XT:

### Quick Start:
```bash
# 1. Download packages + weights locally (internet required)
python quick_setup.py --full

# 2. Upload to Kaggle datasets
kaggle datasets create -p ./offline_setup/wheels --public --dir-mode zip
kaggle datasets create -p ./offline_setup/weights --public --dir-mode zip

# 3. Copy notebook code + configure
# Copy: notebooks/02_train_kaggle_offline_v2.py → Kaggle notebook
# Set: CONFIG['MODEL'] = 'MIT-B5' (or Swin-L, ConvNeXt-L)

# 4. Run all cells (no internet needed!)
```

### Supported Models:
- **MIT-B2** - Fast, 65-70% mIoU (~42 hrs) 
- **MIT-B5** ⭐ - SOTA, 72-75% mIoU (~90 hrs) **RECOMMENDED**
- **Swin-L** - ViT, 73-76% mIoU (~133 hrs)
- **ConvNeXt-L** - Modern CNN, 72-75% mIoU (~117 hrs)

### Complete Guide:
See [OFFLINE_SETUP_GUIDE.md](OFFLINE_SETUP_GUIDE.md) for detailed instructions.

📋 **Quick tools:**
- `quick_setup.py --status` - Check preparation status
- `OFFLINE_CHECKLIST.md` - Step-by-step checklist
- `QUICK_REFERENCE.md` - Common commands

## �📁 Cấu trúc

```
configs/          # YAML configs (dataset, training)
scripts/          # Download & explore utilities
src/data/         # Dataset, augmentations, splits
src/models/       # Model architectures (U-Net, HRNet, ...)
src/training/     # Trainer, losses, metrics
src/evaluation/   # Visualization, analysis
notebooks/        # Kaggle/Jupyter notebooks
outputs/          # Checkpoints, logs, predictions (gitignored)
```

## 🏷️ Classes (11)

| ID | Class | Color |
|----|-------|-------|
| 0 | Sky | ![#00FFFF](https://placehold.co/15x15/00FFFF/00FFFF.png) |
| 1 | Deciduous trees | ![#007F00](https://placehold.co/15x15/007F00/007F00.png) |
| 2 | Coniferous trees | ![#138445](https://placehold.co/15x15/138445/138445.png) |
| 3 | Fallen trees | ![#003541](https://placehold.co/15x15/003541/003541.png) |
| 4 | Dirt ground | ![#824C00](https://placehold.co/15x15/824C00/824C00.png) |
| 5 | Ground vegetation | ![#98FB98](https://placehold.co/15x15/98FB98/98FB98.png) |
| 6 | Rocks | ![#977EAB](https://placehold.co/15x15/977EAB/977EAB.png) |
| 7 | Building | ![#FA9600](https://placehold.co/15x15/FA9600/FA9600.png) |
| 8 | Fence | ![#73B0C3](https://placehold.co/15x15/73B0C3/73B0C3.png) |
| 9 | Car | ![#7B7B7B](https://placehold.co/15x15/7B7B7B/7B7B7B.png) |
| 10 | Empty | ![#000000](https://placehold.co/15x15/000000/000000.png) |

## 📊 Sequence Matrix

| | 0° | -60° | -90° |
|-----|------|------|------|
| **30m** | seq1 | seq2 | seq3 |
| **50m** | seq4 | seq5 | seq6 |
| **80m** | seq7 | seq8 | seq9 |
