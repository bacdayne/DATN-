# Tiền xử lý ảnh & Trả lời câu hỏi bảo vệ

## 1. Vị trí trong pipeline

```
Ảnh hóa đơn
   │
   ▼
[ TIỀN XỬ LÝ ẢNH ]  ← pipeline/preprocess.py  (bật bằng cờ --preprocess)
   │   grayscale → deskew → denoise → CLAHE → upscale → (binarize: tùy chọn)
   ▼
PaddleOCR (nhận dạng chữ + tọa độ)
   │
   ▼
LLM trích xuất trường (Qwen2 fine-tuned)
   │
   ▼
JSON kết quả
```

Cách chạy:
```bash
python run_ocr.py --split test --limit 20 --preprocess
```
Ảnh gốc vẫn được giữ ở `ocr/images/` để hiển thị; ảnh đã xử lý lưu ở `ocr/preprocessed/`.

## 2. Các bước tiền xử lý và lý do

| Bước | Kỹ thuật | Xử lý vấn đề gì |
|------|----------|------------------|
| Grayscale | `cvtColor` | Bỏ nhiễu màu, giảm dữ liệu thừa |
| Deskew | `minAreaRect` + `warpAffine` | Ảnh **chụp nghiêng** → xoay thẳng |
| Denoise | `fastNlMeansDenoising` | Ảnh **nhiễu hạt**, scan kém |
| CLAHE | `createCLAHE` | Ảnh **thiếu sáng / tương phản thấp**, chữ mờ |
| Upscale | `resize INTER_CUBIC` | Ảnh **độ phân giải thấp**, chữ quá nhỏ |
| Binarize (tùy chọn) | `adaptiveThreshold` | Nền lem, bóng — mặc định TẮT |

## 3. Câu trả lời khi thầy cô hỏi

**H: "Bài không có bước tiền xử lý à?"**
> Hệ thống có tiền xử lý ở 2 mức. Thứ nhất, PaddleOCR đã tích hợp sẵn chuẩn hóa ảnh và
> phân loại góc (`use_angle_cls=True`) để tự sửa hướng chữ. Thứ hai, em bổ sung một module
> tiền xử lý tường minh (`pipeline/preprocess.py`) gồm khử nghiêng, khử nhiễu, cân bằng
> tương phản CLAHE và phóng to ảnh, kích hoạt qua cờ `--preprocess`, dành cho ảnh chất lượng thấp.

**H: "Hóa đơn mờ / chụp nghiêng thì xử lý sao?"**
> Ảnh mờ được tăng nét bằng CLAHE (cân bằng tương phản cục bộ) và phóng to nội suy bicubic;
> ảnh nghiêng được nắn thẳng bằng deskew dựa trên `minAreaRect` của khối văn bản. Nhờ đó
> OCR đọc ổn định hơn thay vì phụ thuộc hoàn toàn vào chất lượng ảnh đầu vào.

**H: "Vì sao mặc định TẮT nhị phân hóa (threshold)?"**
> Vì PaddleOCR là OCR học sâu, được huấn luyện trên ảnh xám/màu; nhị phân hóa (kiểu Tesseract cổ điển)
> dễ làm mất nét chữ mảnh và **giảm** độ chính xác. Em để nó là tùy chọn, chỉ bật khi nền bị lem nặng.

**H: "Dataset đã sạch thì cần tiền xử lý không?"**
> Với CORD-v2 (ảnh khá sạch), tiền xử lý mạnh gần như không đổi kết quả. Giá trị của module thể
> hiện rõ khi gặp ảnh thực tế kém chất lượng — đảm bảo hệ thống **bền vững (robust)** khi triển khai thật.

## 4. Kết quả thực nghiệm (CORD-v2, 100 hóa đơn test)

Cấu hình đã chọn: **deskew + CLAHE + upscale** (KHÔNG denoise, KHÔNG binarize).

| Chỉ số | KHÔNG tiền xử lý | CÓ tiền xử lý | Chênh |
|--------|------------------|----------------|-------|
| Instance accuracy | 79.19% | **79.73%** | ↑ +0.54 |
| Field precision | 0.7084 | **0.7126** | ↑ +0.0042 |
| Field recall | 0.7101 | **0.7128** | ↑ +0.0027 |
| Field F1 | 0.7082 | **0.7118** | ↑ +0.0036 |

**Ablation (vì sao bỏ denoise):** cấu hình đầy đủ có denoise chỉ đạt 76.85% (giảm),
vì khử nhiễu làm mượt nét chữ số trên ảnh vốn đã sạch. Bỏ denoise → tăng lên 79.73%.
Bỏ CLAHE (chỉ deskew+upscale) tụt còn 76.33% → CLAHE là bước đóng góp nhiều nhất.

## 5. Điểm cần trung thực khi trình bày
- Kết quả trên CORD-v2 tăng **nhẹ nhưng đồng đều cả 4 chỉ số** (dữ liệu vốn đã sạch nên biên độ nhỏ).
- Giá trị thật của tiền xử lý thể hiện rõ nhất trên ảnh **mờ/nghiêng/thiếu sáng** — có thể chứng minh
  thêm bằng thí nghiệm trên ảnh bị làm suy giảm chất lượng nhân tạo.
