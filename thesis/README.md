# Báo cáo tổng thể dự án (LaTeX)

Báo cáo phân đoạn ngữ nghĩa ảnh UAV rừng + nghiên cứu chiến lược xử lý mất cân bằng lớp.

## Biên dịch

> ⚠️ **Phải dùng XeLaTeX** (có tiếng Việt), không dùng pdfLaTeX.

```powershell
# Trong thư mục thesis/
xelatex main.tex
xelatex main.tex   # chạy 2–3 lần để cập nhật mục lục & tham chiếu chéo
```

- **Overleaf:** Menu → Compiler → **XeLaTeX**. Nếu báo lỗi thiếu font *Times New Roman*,
  mở `main.tex` và đổi `\setmainfont{Times New Roman}` → `\setmainfont{TeX Gyre Termes}`.
- **Font:** mặc định Times New Roman 13pt (chuẩn báo cáo VN). Máy Windows có sẵn font này.

## Cấu trúc

```
thesis/
├── main.tex                 # file gốc — \input các chương
├── refs.bib                 # (tùy chọn) dùng với biblatex
├── figures/                 # nơi đặt hình (logo, biểu đồ, ảnh định tính)
└── tex/
    ├── bia.tex              # bìa (khung xanh + logo) — đã điền thông tin
    ├── loi_cam_doan.tex
    ├── loi_cam_on.tex
    ├── tom_tat.tex
    ├── danh_muc_viet_tat.tex
    ├── ch1_mo_dau.tex
    ├── ch2_tong_quan.tex    # cơ sở lý thuyết + công thức loss
    ├── ch3_du_lieu_phuong_phap.tex
    ├── ch4_ket_qua.tex      # số liệu thật 4 thí nghiệm + phần TODO
    ├── ch5_thao_luan.tex
    ├── ch6_ket_luan.tex
    ├── tham_khao.tex
    └── phu_luc.tex
```

## Cách tìm phần cần điền

Mọi chỗ chưa hoàn thiện được đánh dấu bằng lệnh `\TBD{...}` (hiển thị **đỏ** trong PDF).
Tìm nhanh:

```powershell
Select-String -Path tex\*.tex -Pattern "\\TBD"
```

Các nhóm cần bổ sung chính:
- **Ch.4**: đánh giá thời tiết (sunny/overcast), WildUAV, đa kiến trúc, multi-seed (seed 42, 123).
- **Hình còn lại** (`\HINH{...}`): đường cong huấn luyện (xuất từ CSV) và ảnh so sánh
  định tính 4 thí nghiệm.
- **Lời cảm ơn** (`loi_cam_on.tex`): viết nội dung.

> Đã có sẵn: bìa + logo + lời cam đoan (mẫu Khoa Toán - Tin học); sơ đồ encoder–decoder
> (TikZ); 4 hình thật trong `figures/` (qualitative, per-class metrics, confusion matrix +
> miss-rate, tổng hợp Exp4); related work rừng có trích dẫn.

## Số liệu đã có (từ `notebooks/executed/`)

| Exp | Loss | Test mIoU |
|-----|------|-----------|
| 1 | CE thuần (baseline) | 0.8187 |
| 2 | CE + trọng số lớp | 0.8142 |
| 3 | Focal+Dice+CE (cố định) | 0.8270 |
| 4 | Focal+Dice+CE (thích nghi, seed 2025) | **0.8365** |
