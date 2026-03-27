git push origin main# Git Workflow - Forest Inspection Project

Quy trình làm việc nhóm với Git cho dự án semantic segmentation.

---

## 📌 Branch Structure

```
main          ← Production-ready code, chỉ merge qua PR
├── develop   ← Integration branch, merge features vào đây
├── feature/* ← Tính năng mới
├── fix/*     ← Sửa bug
└── exp/*     ← Thử nghiệm model/training
```

### Branch Naming

| Prefix | Dùng khi | Ví dụ |
|--------|----------|-------|
| `feature/` | Thêm tính năng mới | `feature/data-augmentation` |
| `fix/` | Sửa bug | `fix/label-color-mapping` |
| `exp/` | Thử nghiệm training/model | `exp/hrnet-w48-training` |
| `docs/` | Cập nhật docs | `docs/update-readme` |

---

## 🔄 Quy trình làm việc

### 1. Bắt đầu task mới

```bash
# Luôn bắt đầu từ develop mới nhất
git checkout develop
git pull origin develop

# Tạo branch mới
git checkout -b feature/ten-tinh-nang
```

### 2. Làm việc & commit

```bash
# Xem thay đổi
git status
git diff

# Stage files
git add src/models/unet.py
# hoặc stage tất cả
git add .

# Commit (viết message rõ ràng)
git commit -m "feat: add U-Net model with ResNet encoder"
```

### Commit Message Convention

```
<type>: <mô tả ngắn>

Ví dụ:
feat: add HRNet model with FPN decoder
fix: correct label color mapping for Fallen_trees
exp: train U-Net resnet50 with dice loss
data: add augmentation pipeline
docs: update README with training instructions
refactor: extract metrics to separate module
```

| Type | Ý nghĩa |
|------|---------|
| `feat` | Tính năng mới |
| `fix` | Sửa bug |
| `exp` | Thử nghiệm, kết quả training |
| `data` | Thay đổi data pipeline |
| `docs` | Documentation |
| `refactor` | Refactor code, không đổi logic |
| `test` | Thêm/sửa tests |

### 3. Push lên remote

```bash
# Push branch lên (lần đầu)
git push -u origin feature/ten-tinh-nang

# Các lần sau
git push
```

### 4. Tạo Pull Request (PR)

1. Vào GitHub → **"New Pull Request"**
2. Base: `develop` ← Compare: `feature/ten-tinh-nang`
3. Điền mô tả:
   - **What**: Thay đổi gì
   - **Why**: Tại sao cần thay đổi
   - **How to test**: Cách verify
4. Assign reviewer → Chờ review

### 5. Code Review & Merge

- Reviewer kiểm tra code, comment nếu cần thay đổi
- Author sửa theo feedback, push thêm commits
- Khi approved → **Squash and Merge** vào `develop`
- Xóa branch sau khi merge

### 6. Release (develop → main)

```bash
# Khi develop ổn định, merge vào main
git checkout main
git pull origin main
git merge develop
git push origin main

# Tag version
git tag -a v1.0.0 -m "First training pipeline release"
git push origin v1.0.0
```

---

## ⚠️ Xử lý Conflict

```bash
# Cập nhật develop mới nhất vào branch
git checkout feature/ten-tinh-nang
git fetch origin
git rebase origin/develop

# Nếu conflict:
# 1. Mở file conflict, sửa thủ công
# 2. git add <file-đã-sửa>
# 3. git rebase --continue

# Nếu muốn hủy rebase
git rebase --abort
```

---

## 📋 Checklist trước khi tạo PR

- [ ] Code chạy không lỗi
- [ ] Đã test trên ít nhất 1 sequence
- [ ] Không commit file data lớn (kiểm tra `.gitignore`)
- [ ] Commit messages rõ ràng
- [ ] Cập nhật README nếu thêm tính năng mới

---

## 🧪 Quy trình Training Experiment

Khi thử nghiệm training, dùng branch `exp/`:

```bash
git checkout -b exp/unet-resnet50-lr1e4

# Sau khi train xong, commit kết quả (không commit model weights)
git add configs/train_unet_r50.yaml
git add outputs/logs/  # chỉ log nhẹ, không weights
git commit -m "exp: U-Net resnet50 mIoU=0.72 after 50 epochs"

# Push & tạo PR ghi nhận kết quả
git push -u origin exp/unet-resnet50-lr1e4
```

---

## 👥 Phân công (ví dụ)

| Thành viên | Vai trò | Branch thường dùng |
|------------|---------|-------------------|
| Member A | Data pipeline | `feature/data-*` |
| Member B | Model training | `exp/*`, `feature/model-*` |
| Member C | Evaluation | `feature/eval-*` |
| All | Bug fixes | `fix/*` |
