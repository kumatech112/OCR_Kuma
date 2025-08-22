import os, re, unicodedata
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import pytesseract
import pypdfium2 as pdfium

# =========[ ตั้งค่า ]=========
# แก้ path นี้ถ้าติดตั้ง Tesseract คนละที่ (Windows ทั่วไป)
TESS_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if Path(TESS_EXE).exists():
    pytesseract.pytesseract.tesseract_cmd = TESS_EXE

INPUT_DIR   = Path("input")          # โยนไฟล์ .pdf/.png/.jpg ไว้ในโฟลเดอร์นี้
OUTPUT_XLSX = Path("ocr_lines.xlsx") # ไฟล์ผลลัพธ์
DPI         = 400                    # คุณภาพ render PDF → รูป
OCR_LANG    = "tha+eng"              # ไทย+อังกฤษ
PSM_MODE    = "6"                    # อ่านย่อหน้า/หลายบรรทัด
OEM_MODE    = "1"                    # LSTM
TESS_CFG    = f"--oem {OEM_MODE} --psm {PSM_MODE}"

# =========[ ฟังก์ชันหลัก ]=========

def pdf_to_images(pdf_path: Path, dpi: int = 400) -> List[Image.Image]:
    """แปลง PDF ทุกหน้าเป็น PIL.Image ตาม DPI"""
    imgs = []
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    for i in range(len(doc)):
        page = doc[i]
        pil = page.render(scale=scale).to_pil()
        imgs.append(pil.convert("RGB"))
    return imgs

def pil_to_cv(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def deskew_by_osd(pil: Image.Image) -> Image.Image:
    """หมุนรูปให้อยู่แนวตั้งด้วย Tesseract OSD (ง่ายและแม่นสำหรับสแกนเอกสาร)"""
    try:
        osd = pytesseract.image_to_osd(pil, lang=OCR_LANG)
        # หา 'Rotate: xxx' จากผล OSD
        m = re.search(r"Rotate:\s+(\d+)", osd)
        deg = int(m.group(1)) if m else 0
    except Exception:
        deg = 0

    if deg and deg % 360 != 0:
        # หมุนทวนเข็มเพื่อแก้เอียง (Tesseract ให้มุมที่ต้องหมุนกลับ)
        return pil.rotate(360 - deg, expand=True, fillcolor=(255, 255, 255))
    return pil

def preprocess_for_ocr(pil: Image.Image) -> np.ndarray:
    """แปลงเป็นภาพไบนารีเพื่อ OCR"""
    img = pil_to_cv(pil)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ลดนอยส์เล็กน้อย
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    # Otsu threshold
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

def normalize_thai_spaces(s: str) -> str:
    """จัดช่องว่างให้เหมาะกับภาษาไทย/อังกฤษผสม"""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[ \t\u200b]+", " ", s)   # รวมช่องว่าง/zero-width
    return s.strip()

def ocr_lines_from_image(bw: np.ndarray) -> List[Tuple[str, float]]:
    """
    อ่านทีละ 'บรรทัด' จากภาพไบนารี โดยใช้ image_to_data เพื่อ group ตาม line_num
    return: list ของ (ข้อความบรรทัด, ค่า conf เฉลี่ย)
    """
    data = pytesseract.image_to_data(
        bw, lang=OCR_LANG, config=TESS_CFG, output_type=pytesseract.Output.DICT
    )
    n = len(data["text"])
    buckets = {}  # key = (block, par, line) -> {"words":[(x,y,w,h,txt,conf)], "ymin":int}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        # ข้ามกล่องว่าง
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if key not in buckets:
            buckets[key] = {"words": [], "ymin": y}
        buckets[key]["words"].append((x, y, w, h, txt, conf))
        buckets[key]["ymin"] = min(buckets[key]["ymin"], y)

    # เรียงตามตำแหน่งแนวตั้ง
    lines = []
    for key, rec in sorted(buckets.items(), key=lambda kv: kv[1]["ymin"]):
        # เรียงคำตาม x เพื่อให้ลำดับถูก
        words_sorted = sorted(rec["words"], key=lambda t: t[0])
        words = [w[4] for w in words_sorted]
        confs = [w[5] for w in words_sorted if w[5] >= 0]
        text = normalize_thai_spaces(" ".join(words))
        conf_avg = float(np.mean(confs)) if confs else -1.0
        if text:
            lines.append((text, conf_avg))
    return lines

def process_any_file(path: Path) -> List[Tuple[int, str, float]]:
    """
    รองรับ .pdf และรูป (.png/.jpg/.jpeg)
    คืนค่า: list ของ (page_no, line_text, conf_avg)
    """
    results = []
    if path.suffix.lower() == ".pdf":
        pages = pdf_to_images(path, DPI)
        for idx, pil in enumerate(pages, start=1):
            pil = deskew_by_osd(pil)
            bw = preprocess_for_ocr(pil)
            lines = ocr_lines_from_image(bw)
            for (txt, conf) in lines:
                results.append((idx, txt, conf))
    else:
        pil = Image.open(path).convert("RGB")
        pil = deskew_by_osd(pil)
        bw = preprocess_for_ocr(pil)
        lines = ocr_lines_from_image(bw)
        for (txt, conf) in lines:
            results.append((1, txt, conf))
    return results

def main():
    INPUT_DIR.mkdir(exist_ok=True)
    rows = []  # เก็บทุกบรรทัดเป็นแถวเดียว

    for p in sorted(INPUT_DIR.iterdir()):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
            continue

        try:
            per_file = process_any_file(p)
            for page_no, text, conf in per_file:
                rows.append({
                    "source_file": p.name,
                    "page": page_no,
                    "line_text": text,
                    "conf_avg": round(conf, 2),
                })
            print(f"[OK] {p.name} -> {len(per_file)} lines")
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")

    if not rows:
        print("ไม่พบผลลัพธ์ กรุณาเช็คไฟล์ในโฟลเดอร์ input")
        return

    df = pd.DataFrame(rows, columns=["source_file", "page", "line_text", "conf_avg"])
    # เขียนเป็น Excel แผ่นเดียว แถวละ 1 บรรทัดตามที่ต้องการ
    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="lines", index=False)
    print(f"บันทึกผลลัพธ์ที่: {OUTPUT_XLSX.resolve()}")

if __name__ == "__main__":
    main()
