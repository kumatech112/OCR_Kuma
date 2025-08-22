# ocr_batch_pdf_to_excel.py
import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pytesseract
from pathlib import Path
from datetime import datetime

# ---- ตั้งค่า Tesseract (แก้พาธตามเครื่องคุณ) ----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESS_LANG = "tha+eng"
TESS_CONFIG = "--psm 6"

INPUT_DIR = Path("Input")
OUTPUT_DIR = Path("Output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def pdf_to_images(pdf_path: Path, dpi=300):
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        yield i, img[:, :, ::-1]  # BGR

def deskew_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # ให้พื้นหลังเป็นขาว ตัวอักษรดำ
    if (bin_img == 255).mean() < 0.5:
        bin_img = cv2.bitwise_not(bin_img)

    coords = np.column_stack(np.where(bin_img == 0))
    if coords.size == 0:
        return bgr, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(bgr, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, float(angle)

def ocr_lines_from_image(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)  # ดึงคอนทราสต์เล็กน้อย
    data = pytesseract.image_to_data(
        gray, lang=TESS_LANG, config=TESS_CONFIG,
        output_type=pytesseract.Output.DICT
    )
    rows, buff = [], []
    n = len(data["text"])
    last_key = None
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if last_key is None:
            last_key = key
        if key != last_key:
            if buff:
                rows.append(" ".join(buff).split())
                rows[-1] = " ".join(rows[-1])
            buff = [txt]
            last_key = key
        else:
            buff.append(txt)
    if buff:
        rows.append(" ".join(buff).split())
        rows[-1] = " ".join(rows[-1])

    # ลบช่องว่างซ้ำ
    rows = [" ".join(r.split()) for r in rows if r and str(r).strip()]
    return rows

def process_all_pdfs():
    records = []
    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print("ไม่พบไฟล์ .pdf ในโฟลเดอร์ Input/")
        return None

    for pdf in pdf_files:
        for page_no, img in pdf_to_images(pdf, dpi=300):
            img_fixed, ang = deskew_bgr(img)
            lines = ocr_lines_from_image(img_fixed)
            for i, line in enumerate(lines, start=1):
                records.append({
                    "file": pdf.name,
                    "page": page_no,
                    "line_no": i,
                    "text": line,
                    "angle_deg": round(ang, 2),
                })
    if not records:
        print("ไม่มีข้อความที่ OCR ได้")
        return None

    df = pd.DataFrame(records, columns=["file","page","line_no","text","angle_deg"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"ocr_result_{ts}.xlsx"
    df.to_excel(out_path, index=False)
    return out_path, len(df), len(pdf_files)

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = process_all_pdfs()
    if result:
        out_path, n_rows, n_files = result
        print(f"สำเร็จ ✓ รวม {n_files} ไฟล์ → {n_rows} แถว")
        print(f"ไฟล์ผลลัพธ์: {out_path}")
