import re
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pytesseract
import pypdfium2 as pdfium
import unicodedata
import math

# ---------- ตั้งค่า ----------
TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # แก้ถ้าคนละที่
if os.path.exists(TESS):
    pytesseract.pytesseract.tesseract_cmd = TESS

INPUT_DIR   = Path("input")
OUTPUT_XLSX = Path("pdf_lines.xlsx")
DPI = 400
OCR_LANG = "tha+eng"
CFG = "--oem 1 --psm 6"   # อ่านย่อหน้า/หลายบรรทัดทั่วไป

# ---------- ฟังก์ชัน ----------
def pdf_to_images(pdf_path: Path, dpi=400):
    imgs = []
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    for i in range(len(doc)):
        page = doc[i]
        pil = page.render(scale=scale).to_pil()
        imgs.append(pil)
    return imgs

import unicodedata
import math

def deskew(bw: np.ndarray) -> np.ndarray:
    # ค้นหาเส้น text baseline ด้วย Hough แล้วหมุนแก้เอียง
    edges = cv2.Canny(bw, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 180)
    if lines is None:
        return bw
    angles = []
    for l in lines[:200]:
        theta = l[0][1]
        deg = (theta*180/np.pi) - 90  # ทำให้ใกล้ 0 สำหรับเส้นแนวนอน
        if -20 < deg < 20:
            angles.append(deg)
    if not angles:
        return bw
    angle = np.median(angles)
    h, w = bw.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def preprocess(pil: Image.Image) -> np.ndarray:
    img = np.array(pil.convert("RGB"))
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ตัดพื้นหลัง/ลายน้ำจาง ๆ ด้วยวิธี "divide by blur"
    blur = cv2.GaussianBlur(g, (0,0), 21)
    norm = cv2.divide(g, blur, scale=255)

    # เพิ่มคอนทราสต์ด้วย CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g2 = clahe.apply(norm)

    # แยกเส้นอักษรด้วย adaptive threshold (ช่วยกับสแกนซีด/ไม่สม่ำเสมอ)
    bw = cv2.adaptiveThreshold(g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)

    # ลบจุดเล็ก ๆ
    bw = cv2.medianBlur(bw, 3)

    # แก้เอียง
    bw = deskew(bw)
    return bw

def _lines_from_tesseract(bw: np.ndarray, config: str):
    data = pytesseract.image_to_data(
        bw, lang=OCR_LANG, config=config, output_type=pytesseract.Output.DICT
    )
    n = len(data["text"])
    buckets = {}  # group เป็นบรรทัด
    conf_all = []
    for i in range(n):
        # robust กับ conf ที่เป็น str/int
        try: conf = int(data["conf"][i])
        except (ValueError, TypeError): conf = -1
        txt = (data["text"][i] or "").strip()
        if not txt: 
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        y = data["top"][i]; x = data["left"][i]
        if key not in buckets: buckets[key] = {"y": y, "words": []}
        buckets[key]["y"] = min(buckets[key]["y"], y)
        buckets[key]["words"].append((x, txt, conf))
        if conf >= 0: conf_all.append(conf)

    ordered = sorted(buckets.values(), key=lambda d: d["y"])
    lines = []
    for ln in ordered:
        ws = sorted(ln["words"], key=lambda t: t[0])
        # คงช่องไฟให้ติดกันมากขึ้น
        words = [w for _, w, _ in ws]
        text = " ".join(words)
        text = unicodedata.normalize("NFC", text)  # รวมสระ/วรรณยุกต์ให้ถูกต้อง
        # ลดช่องว่างซ้ำ
        text = re.sub(r"\s{2,}", " ", text).strip()
        if text:
            lines.append(text)

    avg_conf = float(np.mean(conf_all)) if conf_all else -1.0
    return lines, avg_conf

def ocr_lines(pil: Image.Image):
    bw = preprocess(pil)
    # ลองหลายโหมด: 3=auto, 4=คอลัมน์, 6=บรรทัดทั่วไป, 11/12=single text line/block
    configs = [
        "--oem 1 --psm 4 -c preserve_interword_spaces=1",
        "--oem 1 --psm 3 -c preserve_interword_spaces=1",
        "--oem 1 --psm 6 -c preserve_interword_spaces=1",
        "--oem 1 --psm 11 -c preserve_interword_spaces=1",
        "--oem 1 --psm 12 -c preserve_interword_spaces=1",
    ]
    best_lines, best_conf = [], -1
    for cfg in configs:
        lines, c = _lines_from_tesseract(bw, cfg)
        if c > best_conf:
            best_lines, best_conf = lines, c
    return best_lines


# ---------- main ----------
def main():
    if not INPUT_DIR.exists():
        print(f"ไม่พบ {INPUT_DIR.resolve()}"); sys.exit(1)

    pdfs = [p for p in sorted(INPUT_DIR.iterdir()) if p.suffix.lower()==".pdf"]
    if not pdfs:
        print("ไม่พบไฟล์ .pdf ใน input/"); sys.exit(1)

    rows = []
    max_lines = 0

    for pdf in pdfs:
        pages = pdf_to_images(pdf, dpi=DPI)
        for idx, pil in enumerate(pages, start=1):
            lines = ocr_lines(pil)
            max_lines = max(max_lines, len(lines))
            row = {"filename": pdf.name, "page": idx}
            # เก็บบรรทัดที่ 1..N
            for i, t in enumerate(lines, start=1):
                row[f"บรรทัดที่ {i}"] = t
            rows.append(row)
            print(f"OK: {pdf.name} page {idx} → {len(lines)} บรรทัด")

    # ทำให้ทุกแถวมีคอลัมน์เท่ากัน
    cols = ["filename", "page"] + [f"บรรทัดที่ {i}" for i in range(1, max_lines+1)]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns: df[c] = None
    df = df[cols]
    df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
    print(f"เสร็จแล้ว → {OUTPUT_XLSX.resolve()}")

if __name__ == "__main__":
    main()
