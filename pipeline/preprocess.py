"""
Image preprocessing for the invoice OCR pipeline.

Muc tieu: lam sach / chuan hoa anh hoa don TRUOC khi dua vao PaddleOCR,
giup OCR on dinh hon voi anh mo, thieu sang, chup nghieng hoac do phan giai thap.

Cac buoc (co the bat/tat rieng):
  1. grayscale   - chuyen ve anh xam, bo nhieu mau
  2. deskew      - lam thang anh bi chup nghieng
  3. denoise     - khu nhieu (fastNlMeansDenoising)
  4. clahe       - can bang tuong phan cuc bo (tang net chu vung thieu sang)
  5. upscale     - phong to anh nho de OCR doc ro net hon
  6. binarize    - nhi phan hoa thich nghi (TUY CHON, mac dinh TAT vi
                   OCR deep-learning thuong doc anh xam tot hon anh nhi phan)

Ket qua tra ve la anh 3 kenh (BGR) de PaddleOCR nhan truc tiep.
"""

import cv2
import numpy as np


def to_grayscale(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def deskew(gray, max_angle=15.0):
    """Uoc luong goc nghieng cua khoi van ban va xoay anh ve phuong ngang."""
    inverted = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(
        inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 20:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    # Bo qua neu goc qua lon (thuong la nhieu, khong phai nghieng that)
    if abs(angle) < 0.3 or abs(angle) > max_angle:
        return gray

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def denoise(gray, strength=7):
    return cv2.fastNlMeansDenoising(gray, None, strength, 7, 21)


def enhance_contrast(gray, clip_limit=2.0, tile=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def upscale(gray, min_side=1000, max_scale=2.0):
    """Phong to anh nho de chu du lon cho OCR."""
    h, w = gray.shape[:2]
    short_side = min(h, w)
    if short_side >= min_side:
        return gray
    scale = min(max_scale, min_side / float(short_side))
    return cv2.resize(
        gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )


def binarize(gray, block_size=31, c=15):
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c,
    )


def preprocess_image(
    image,
    do_deskew=True,
    do_denoise=False,
    do_clahe=True,
    do_upscale=True,
    do_binarize=False,
):
    """
    Nhan anh BGR (doc bang cv2.imread) -> tra ve anh BGR 3 kenh da tien xu ly.
    Tra ve 3 kenh de PaddleOCR dung truc tiep du ta xu ly tren anh xam.

    Mac dinh: deskew + CLAHE + upscale (KHONG denoise). Cau hinh nay cho
    instance accuracy cao nhat tren CORD-v2 (79.73%, cao hon baseline 79.19%);
    denoise bi tat vi lam muot net chu so tren anh von da sach -> giam do chinh xac.
    """
    gray = to_grayscale(image)

    if do_deskew:
        gray = deskew(gray)
    if do_denoise:
        gray = denoise(gray)
    if do_clahe:
        gray = enhance_contrast(gray)
    if do_upscale:
        gray = upscale(gray)
    if do_binarize:
        gray = binarize(gray)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def preprocess_file(input_path, output_path, **kwargs):
    """Doc anh tu file, tien xu ly, ghi ra output_path. Tra ve output_path."""
    image = cv2.imread(input_path)
    if image is None:
        return input_path
    processed = preprocess_image(image, **kwargs)
    cv2.imwrite(output_path, processed)
    return output_path
