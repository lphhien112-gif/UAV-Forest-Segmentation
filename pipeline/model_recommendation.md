# Model & Encoder Recommendation — UAV Forest Segmentation

**Mục tiêu:** Q2 journal | **GPU:** RTX 6000 Pro 48GB | **Thư viện:** segmentation_models_pytorch (smp)

---

## Chiến lược thí nghiệm cho paper

Một paper Q2 cần **so sánh có hệ thống**: CNN cổ điển → CNN hiện đại → Transformer → Hybrid. Dưới đây là **12 experiments** được chia thành 4 nhóm.

---

## Group A: CNN Baselines (phải có để so sánh)

| # | MODEL_NAME | ENCODER | Params | VRAM ~est | Vai trò trong paper |
|---|---|---|---|---|---|
| A1 | `unet` | `resnet34` | ~24M | ~4GB | Baseline kinh điển |
| A2 | `unet` | `resnet101` | ~44M | ~8GB | CNN sâu hơn |
| A3 | `deeplabv3plus` | `resnet101` | ~60M | ~10GB | ASPP + multi-scale, rất phổ biến trong RS papers |

```python
# A1
smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=11)
# A2
smp.Unet(encoder_name='resnet101', encoder_weights='imagenet', classes=11)
# A3
smp.DeepLabV3Plus(encoder_name='resnet101', encoder_weights='imagenet', classes=11)
```

> [!NOTE]
> ResNet + DeepLabV3+ là baseline bắt buộc — hầu hết reviewer sẽ hỏi nếu thiếu.

---

## Group B: Modern CNN (EfficientNet, ConvNeXt)

| # | MODEL_NAME | ENCODER | Params | VRAM ~est | Vai trò |
|---|---|---|---|---|---|
| B1 | `unet` | `efficientnet-b5` | ~30M | ~8GB | Efficient scaling |
| B2 | `upernet` | `tu-convnext_base` | ~90M | ~16GB | ⭐ CNN hiện đại nhất (2022), ngang Transformer |
| B3 | `deeplabv3plus` | `tu-convnext_small` | ~50M | ~12GB | ConvNeXt + multi-scale |

```python
# B1
smp.Unet(encoder_name='efficientnet-b5', encoder_weights='imagenet', classes=11)
# B2 - ⭐ Rất mạnh
smp.UPerNet(encoder_name='tu-convnext_base', encoder_weights='imagenet', classes=11)
# B3
smp.DeepLabV3Plus(encoder_name='tu-convnext_small', encoder_weights='imagenet', classes=11)
```

> [!TIP]
> ConvNeXt (2022) là CNN thuần nhưng thiết kế theo Transformer → kết quả ngang Swin Transformer. Paper có thể argue "modern CNN vs Transformer".

---

## Group C: Transformer-based (điểm nhấn của paper)

| # | MODEL_NAME | ENCODER | Params | VRAM ~est | Vai trò |
|---|---|---|---|---|---|
| C1 | `segformer` | `mit_b2` | ~28M | ~8GB | SegFormer nhẹ |
| C2 | `segformer` | `mit_b5` | ~85M | ~18GB | ⭐ SegFormer lớn nhất, SOTA |
| C3 | `upernet` | `tu-swinv2_small_window16_256` | ~70M | ~16GB | ⭐ Swin Transformer V2 |

```python
# C1
smp.Segformer(encoder_name='mit_b2', encoder_weights='imagenet', classes=11)
# C2 - ⭐ Kỳ vọng kết quả tốt nhất
smp.Segformer(encoder_name='mit_b5', encoder_weights='imagenet', classes=11)
# C3 - ⭐ Rất mạnh
smp.UPerNet(encoder_name='tu-swinv2_small_window16_256', encoder_weights='imagenet', classes=11)
```

> [!IMPORTANT]
> **SegFormer mit_b5** và **UPerNet + Swin V2** là 2 model kỳ vọng mIoU cao nhất. Đây sẽ là điểm nhấn của paper.

---

## Group D: Ablation — cùng encoder, khác decoder

| # | MODEL_NAME | ENCODER | Mục đích |
|---|---|---|---|
| D1 | `unet` | `mit_b3` | UNet decoder + Transformer encoder |
| D2 | `deeplabv3plus` | `mit_b3` | DeepLab decoder + Transformer encoder |
| D3 | `segformer` | `mit_b3` | SegFormer decoder + cùng encoder |

```python
# Cùng encoder mit_b3, khác decoder → ablation study
smp.Unet(encoder_name='mit_b3', encoder_weights='imagenet', classes=11)
smp.DeepLabV3Plus(encoder_name='mit_b3', encoder_weights='imagenet', classes=11)
smp.Segformer(encoder_name='mit_b3', encoder_weights='imagenet', classes=11)
```

> [!TIP]
> Group D cho phép viết 1 section "Impact of decoder architecture" — rất có giá trị cho reviewer.

---

## Tóm tắt: 12 experiments

| # | Config | Loại | Ưu tiên |
|---|---|---|---|
| A1 | UNet + ResNet34 | CNN baseline | 🔴 Bắt buộc |
| A2 | UNet + ResNet101 | CNN sâu | 🟡 Nên có |
| A3 | DeepLabV3+ + ResNet101 | CNN multi-scale | 🔴 Bắt buộc |
| B1 | UNet + EfficientNet-B5 | Efficient CNN | 🟡 Nên có |
| B2 | UPerNet + ConvNeXt-B | Modern CNN | 🔴 Quan trọng |
| B3 | DeepLabV3+ + ConvNeXt-S | Modern CNN | 🟢 Tùy chọn |
| C1 | SegFormer + MiT-B2 | Transformer nhẹ | 🟡 Nên có |
| C2 | SegFormer + MiT-B5 | Transformer lớn | 🔴 **Quan trọng nhất** |
| C3 | UPerNet + Swin V2-S | Transformer | 🔴 Quan trọng |
| D1 | UNet + MiT-B3 | Ablation | 🟡 Nên có |
| D2 | DeepLabV3+ + MiT-B3 | Ablation | 🟡 Nên có |
| D3 | SegFormer + MiT-B3 | Ablation | 🟡 Nên có |

---

## Training Config khuyến nghị (RTX 6000 Pro 48GB)

```python
IMG_SIZE    = (512, 512)   # hoặc (768, 768) nếu muốn resolution cao hơn
BATCH_SIZE  = 16           # 48GB cho phép batch lớn
EPOCHS      = 100          # paper cần train đủ lâu
LR          = 1e-4         # Transformer thường dùng LR nhỏ hơn CNN
USE_AMP     = True         # mixed precision
```

| Hyperparameter | CNN (Group A) | Transformer (Group B-D) |
|---|---|---|
| Learning Rate | `1e-3` | `6e-5` đến `1e-4` |
| Batch Size | `16–32` | `8–16` |
| Weight Decay | `1e-4` | `0.01` |
| Warmup | Không cần | 5–10 epochs |
| Scheduler | CosineAnnealing | CosineAnnealing hoặc PolyLR |

---

## Pretrained Weights cần download

Tất cả weights tự download khi đặt `encoder_weights='imagenet'`. Các file chính:

| Encoder | Source | Size |
|---|---|---|
| `resnet34` | torchvision | ~87MB |
| `resnet101` | torchvision | ~171MB |
| `efficientnet-b5` | timm | ~120MB |
| `mit_b2` | smp/timm | ~100MB |
| `mit_b3` | smp/timm | ~180MB |
| `mit_b5` | smp/timm | ~330MB |
| `tu-convnext_base` | timm | ~350MB |
| `tu-convnext_small` | timm | ~200MB |
| `tu-swinv2_small_window16_256` | timm | ~200MB |

> [!WARNING]
> Nếu train offline (không có internet), cần download weights trước và load thủ công.
> Dùng notebook `01_prepare_offline_assets.ipynb` để download tất cả.

---

## Metrics cần báo cáo (chuẩn Q2 journal)

| Metric | Ý nghĩa |
|---|---|
| **mIoU** | Mean Intersection over Union — metric chính |
| **Pixel Accuracy** | Tổng pixel đúng |
| **Per-class IoU** | IoU từng class — bảng chi tiết |
| **FLOPs** | Computational cost |
| **Params** | Model size |
| **FPS** | Inference speed |

---

## Cấu trúc bảng trong paper

### Table: Comparison of segmentation architectures on Forest Inspection dataset

| Method | Encoder | Params (M) | FLOPs (G) | Sky | Dec. Trees | Con. Trees | ... | mIoU | PA |
|---|---|---|---|---|---|---|---|---|---|
| U-Net | ResNet-34 | 24.4 | 38.2 | 0.xx | 0.xx | 0.xx | ... | 0.xx | 0.xx |
| U-Net | ResNet-101 | 44.1 | 72.5 | ... | ... | ... | ... | ... | ... |
| DeepLabV3+ | ResNet-101 | 59.6 | 89.3 | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| **SegFormer** | **MiT-B5** | **84.7** | **xx** | ... | ... | ... | ... | **0.xx** | **0.xx** |

---

## Thứ tự chạy khuyến nghị

```
Tuần 1: A1 (UNet+R34) → A3 (DLV3++R101) → C1 (SegFormer+B2)
         → báo cáo sơ bộ cho thầy

Tuần 2: C2 (SegFormer+B5) → C3 (UPerNet+SwinV2) → B2 (UPerNet+ConvNeXt)
         → xác định model tốt nhất

Tuần 3: D1-D3 (ablation) → A2, B1, B3 (bổ sung)
         → hoàn thiện bảng so sánh

Tuần 4: Cải thiện model tốt nhất (augmentation, loss, post-processing)
         → kết quả cuối cùng cho paper
```
