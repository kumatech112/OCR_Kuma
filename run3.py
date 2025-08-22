# ocr_pdf_input_to_output.py
import re, os, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pytesseract
import pypdfium2 as pdfium
import unicodedata

# ---------- ตั้งค่า ----------
TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # แก้ถ้าอยู่คนละที่
if os.path.exists(TESS):
    pytesseract.pytesseract.tesseract_cmd = TESS

INPUT_DIR = Path("Output")
OUTPUT_DIR = Path("Folder")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
OCR_LANG = "tha+eng"
# จะลองหลาย PSM แล้วเลือกที่ค่า conf โดยเฉลี่ยดีที่สุด
PSM_LIST = ["--psm 4", "--psm 3", "--psm 6", "--psm 11", "--psm 12"]
OEM = "--oem 1"

# ---------- ฟังก์ชัน ----------
def pdf_to_images(pdf_path: Path, dpi=300):
    """แปลง PDF → PIL images (หนึ่งหน้า/รูป) ด้วย pypdfium2"""
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    for i in range(len(doc)):
        page = doc[i]
        pil = page.render(scale=scale).to_pil()  # RGB PIL
        yield pil

def deskew(bw: np.ndarray):
    """
    ค้นหาแนวเส้นด้วย Hough แล้วหมุนแก้เอียง
    return: (bw_rotated, angle_deg)
    """
    edges = cv2.Canny(bw, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 180)
    if lines is None:
        return bw, 0.0
    angles = []
    for l in lines[:200]:
        theta = l[0][1]
        deg = (theta*180/np.pi) - 90  # ใกล้ศูนย์สำหรับแนวนอน
        if -20 < deg < 20:
            angles.append(deg)
    if not angles:
        return bw, 0.0
    angle = float(np.median(angles))
    h, w = bw.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

def preprocess(pil: Image.Image):
    """
    ทำความสะอาดภาพ + แปลงเป็นขาวดำ + แก้เอียง
    return: (bw, angle_deg)
    """
    img = np.array(pil.convert("RGB"))
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ล้างพื้นหลัง/เงา ด้วย divide-by-blur
    blur = cv2.GaussianBlur(g, (0, 0), 21)
    norm = cv2.divide(g, blur, scale=255)

    # เพิ่มคอนทราสต์
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2 = clahe.apply(norm)

    # แยกเส้นอักษร
    bw = cv2.adaptiveThreshold(g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)

    # ลบ noise เล็ก ๆ
    bw = cv2.medianBlur(bw, 3)

    # แก้เอียง
    bw, angle = deskew(bw)
    return bw, angle

THAI_RUN = re.compile(r"([\u0E00-\u0E7F])\s+([\u0E00-\u0E7F])")
def fix_thai_spacing(s: str) -> str:
    """ลบช่องไฟแปลก ๆ ระหว่างตัวอักษรไทย"""
    s = unicodedata.normalize("NFC", s)
    while True:
        new = THAI_RUN.sub(r"\1\2", s)
        if new == s:
            return new
        s = new

def _lines_from_tesseract(bw: np.ndarray, config: str):
    """
    OCR แบบเก็บเป็น 'บรรทัด' ด้วย image_to_data
    return: (list[str] lines, avg_conf)
    """
    data = pytesseract.image_to_data(
        bw, lang=OCR_LANG, config=f"{OEM} {config}", output_type=pytesseract.Output.DICT
    )
    n = len(data["text"])
    buckets = {}  # group ตาม block/paragraph/line
    conf_all = []
    for i in range(n):
        try:
            conf = int(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        y = data["top"][i]
        x = data["left"][i]
        if key not in buckets:
            buckets[key] = {"y": y, "words": []}
        buckets[key]["y"] = min(buckets[key]["y"], y)
        buckets[key]["words"].append((x, txt, conf))
        if conf >= 0:
            conf_all.append(conf)

    ordered = sorted(buckets.values(), key=lambda d: d["y"])
    lines = []
    for ln in ordered:
        ws = sorted(ln["words"], key=lambda t: t[0])  # ซ้าย→ขวา
        words = [w for _, w, _ in ws]
        text = " ".join(words)
        text = fix_thai_spacing(text)
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        if text:
            lines.append(text)

    avg_conf = float(np.mean(conf_all)) if conf_all else -1.0
    return lines, avg_conf

def ocr_lines(pil: Image.Image):
    """
    คืน: (best_lines, angle_deg, best_conf)
    """
    bw, angle = preprocess(pil)
    best_lines, best_conf = [], -1.0
    for psm in PSM_LIST:
        lines, c = _lines_from_tesseract(bw, psm)
        if c > best_conf:
            best_lines, best_conf = lines, c
    return best_lines, angle, best_conf

# ---------- main ----------
def main():
    if not INPUT_DIR.exists():
        print(f"ไม่พบโฟลเดอร์: {INPUT_DIR.resolve()}")
        sys.exit(1)

    pdfs = sorted([p for p in INPUT_DIR.glob("*.pdf")])
    if not pdfs:
        print("ไม่พบไฟล์ .pdf ในโฟลเดอร์ Input/")
        sys.exit(1)

    records = []
    for pdf in pdfs:
        for page_idx, pil in enumerate(pdf_to_images(pdf, dpi=DPI), start=1):
            lines, angle, conf = ocr_lines(pil)
            for i, t in enumerate(lines, start=1):
                records.append({
                    "file": pdf.name,
                    "page": page_idx,
                    "line_no": i,
                    "text": t,
                    "angle_deg": round(float(angle), 2),
                    "avg_conf": round(float(conf), 1)
                })
            print(f"OK: {pdf.name} page {page_idx} → {len(lines)} บรรทัด (rot {angle:.2f}°, conf {conf:.1f})")

    if not records:
        print("ไม่มีข้อความที่ OCR ได้"); sys.exit(0)

    df = pd.DataFrame(records, columns=["file","page","line_no","text","angle_deg","avg_conf"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"ocr_result_{ts}.xlsx"
    df.to_excel(out_path, index=False)
    print(f"เสร็จแล้ว → {out_path.resolve()}  (ทั้งหมด {len(df)} แถว, {len(pdfs)} ไฟล์)")

if __name__ == "__main__":
    main()
