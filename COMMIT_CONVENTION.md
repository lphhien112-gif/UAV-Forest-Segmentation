# 📝 Commit Message Convention

> Hướng dẫn viết commit message chuẩn cho dự án, dựa trên [Conventional Commits](https://www.conventionalcommits.org/) v1.0.0.

---

## Cấu trúc Commit Message

```
<type>(<scope>): <subject>

[body]

[footer]
```

| Thành phần | Bắt buộc | Mô tả |
|------------|----------|-------|
| `type` | ✅ | Loại thay đổi (xem bảng bên dưới) |
| `scope` | ❌ | Phạm vi ảnh hưởng (module, file, feature) |
| `subject` | ✅ | Mô tả ngắn gọn thay đổi (≤ 72 ký tự) |
| `body` | ❌ | Giải thích chi tiết lý do & cách thay đổi |
| `footer` | ❌ | Breaking changes, issue references |

---

## Các loại Type

### Core Types

| Type | Emoji | Mô tả | Ví dụ |
|------|-------|--------|-------|
| `feat` | ✨ | Thêm tính năng mới | `feat(model): add MixVisionTransformer backbone` |
| `fix` | 🐛 | Sửa lỗi | `fix(train): resolve NaN loss on mixed precision` |
| `docs` | 📚 | Thay đổi tài liệu | `docs: update README with training instructions` |
| `style` | 💄 | Format code (không thay đổi logic) | `style: apply black formatter to utils/` |
| `refactor` | ♻️ | Tái cấu trúc code (không fix bug, không thêm feature) | `refactor(dataset): simplify augmentation pipeline` |
| `perf` | ⚡ | Cải thiện hiệu năng | `perf(dataloader): enable pin_memory and prefetch` |
| `test` | ✅ | Thêm hoặc sửa test | `test(metrics): add unit test for mIoU calculation` |
| `build` | 📦 | Thay đổi build system, dependencies | `build: upgrade segmentation-models-pytorch to 0.4` |
| `ci` | 🔧 | Thay đổi CI/CD config | `ci: add GitHub Actions for linting` |
| `chore` | 🔨 | Công việc phụ trợ, maintenance | `chore: clean up unused checkpoint files` |
| `revert` | ⏪ | Hoàn tác commit trước đó | `revert: revert feat(model) added in abc1234` |

### Extended Types (Research / ML Projects)

| Type | Emoji | Mô tả | Ví dụ |
|------|-------|--------|-------|
| `experiment` | 🧪 | Thêm/sửa thí nghiệm | `experiment(seg): run UNet++ with MIT-B5 backbone` |
| `data` | 🗃️ | Thay đổi liên quan đến dataset | `data: add seq9 test split from Zenodo` |
| `model` | 🧠 | Thay đổi kiến trúc model | `model: implement FPN decoder head` |
| `config` | ⚙️ | Thay đổi cấu hình training/eval | `config: set IMG_SIZE=768 and BATCH_SIZE=4` |
| `notebook` | 📓 | Cập nhật Jupyter notebook | `notebook(eval): refactor 03_evaluate.ipynb` |
| `paper` | 📄 | Thay đổi liên quan đến paper/report | `paper: draft methodology section` |
| `viz` | 📊 | Thêm/sửa visualization | `viz: add confusion matrix heatmap` |

---

## Quy tắc viết Subject

1. **Dùng câu mệnh lệnh** (imperative mood): `add`, `fix`, `update`, `remove` — không dùng `added`, `fixes`, `updating`
2. **Không viết hoa chữ cái đầu** của subject
3. **Không kết thúc bằng dấu chấm** (`.`)
4. **Giới hạn 72 ký tự** cho toàn bộ dòng đầu tiên
5. **Viết bằng tiếng Anh** để đảm bảo tính nhất quán

### ✅ Đúng

```
feat(model): add ResNet-50 encoder for DeepLabV3+
fix(loss): handle edge case when mask is all background
docs(readme): add installation guide for offline setup
```

### ❌ Sai

```
feat(model): Added ResNet-50 encoder.          # past tense + dấu chấm
Fix bug                                         # thiếu type format, quá chung chung
update                                          # không mô tả gì cả
feat: This commit adds a new feature to the...  # quá dài, viết hoa
```

---

## Quy tắc viết Body

- Cách dòng đầu tiên **1 dòng trống**
- Giải thích **tại sao** thay đổi, không chỉ **cái gì** thay đổi
- Mỗi dòng giới hạn **72 ký tự**
- Dùng bullet points nếu có nhiều thay đổi

```
feat(train): implement cosine annealing with warm restarts

- Replace StepLR with CosineAnnealingWarmRestarts scheduler
- Initial learning rate set to 1e-4 with T_0=10 epochs
- This improves convergence stability on the UAV dataset
  compared to fixed step decay (validated on seq8)
```

---

## Quy tắc viết Footer

### Breaking Changes

Dùng prefix `BREAKING CHANGE:` trong footer:

```
refactor(dataset)!: restructure directory layout

BREAKING CHANGE: dataset paths now follow /data/{split}/{seq}/ format.
All existing config files must be updated.
```

### Issue References

```
fix(eval): correct mIoU calculation for ignore class

Closes #42
Refs #38, #41
```

---

## Scope phổ biến trong dự án

| Scope | Phạm vi |
|-------|---------|
| `model` | Kiến trúc model, backbone, decoder |
| `train` | Training pipeline, loss, optimizer |
| `eval` | Evaluation, metrics, inference |
| `data` | Dataset, dataloader, augmentation |
| `config` | Hyperparameters, experiment config |
| `notebook` | Jupyter notebooks |
| `paper` | LaTeX report, figures, tables |
| `utils` | Utility functions, helpers |
| `viz` | Visualization, plotting |
| `deps` | Dependencies, requirements |

---

## Ví dụ hoàn chỉnh

### Simple commit

```
fix(dataloader): set num_workers=0 on Windows to avoid spawn error
```

### Commit với body

```
feat(train): add mixed precision training with GradScaler

- Enable torch.cuda.amp for forward pass
- Use GradScaler to prevent underflow in FP16
- Reduces memory usage by ~40% on RTX 6000 Pro
- Allows batch_size=8 at 768x768 resolution
```

### Commit với breaking change

```
refactor(config)!: migrate from argparse to YAML config files

All training parameters are now defined in YAML files under configs/.
CLI arguments are no longer supported.

BREAKING CHANGE: remove all argparse-based configuration.
Users must create a YAML config file to run training.

Refs #15
```

### Merge / Release commit

```
chore(release): v1.2.0

- feat(model): add UNet++ with MIT-B5 backbone
- feat(train): implement Focal + Dice combined loss
- fix(eval): correct per-class IoU averaging
- perf(data): optimize augmentation pipeline with albumentations
```

---

## Git Hooks (Khuyến nghị)

Cài đặt [commitlint](https://commitlint.js.org/) để tự động kiểm tra format:

```bash
# Cài đặt
npm install --save-dev @commitlint/{config-conventional,cli}

# Tạo config
echo "module.exports = { extends: ['@commitlint/config-conventional'] };" > commitlint.config.js

# Kết hợp với husky
npx husky add .husky/commit-msg 'npx --no -- commitlint --edit "$1"'
```

---

## Quick Reference Card

```
feat:       ✨  New feature
fix:        🐛  Bug fix
docs:       📚  Documentation
style:      💄  Code style (formatting)
refactor:   ♻️  Code refactoring
perf:       ⚡  Performance
test:       ✅  Tests
build:      📦  Build / Dependencies
ci:         🔧  CI/CD
chore:      🔨  Maintenance
revert:     ⏪  Revert changes
experiment: 🧪  ML Experiments
data:       🗃️  Dataset changes
model:      🧠  Model architecture
config:     ⚙️  Configuration
notebook:   📓  Notebooks
paper:      📄  Paper / Report
viz:        📊  Visualization
```

---

> 💡 **Tip**: Commit thường xuyên, mỗi commit chỉ chứa **một thay đổi logic duy nhất**. Điều này giúp dễ dàng review, revert và theo dõi lịch sử dự án.
