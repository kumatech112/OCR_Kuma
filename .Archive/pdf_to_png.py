# --- PDF -> Images (ด้วย pypdfium2) + Deskew + Crop + Resize ---
from pathlib import Path
import cv2, numpy as np
import pypdfium2 as pdfium

PDF_PATH   = Path("input.pdf")
IMG_DIR    = Path("images_raw")
OUT_DIR    = Path("images_clean")
DPI        = 400
FINAL_SIZE = (2480, 3508)   # A4 ที่ ~300dpi
IMG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

def pdf_to_images_pypdfium2(pdf_path: Path, out_dir: Path, dpi=400):
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0  # 1pt = 1/72 inch
    for i in range(len(doc)):
        page = doc[i]
        pil = page.render(scale=scale).to_pil()  # ได้ PIL.Image
        out_file = out_dir / f"page_{i+1:03}.png"
        pil.save(out_file, "PNG")

def deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(bw > 0))
    if coords.size == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h,w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def crop_and_resize(img, size=FINAL_SIZE):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        img = img[y:y+h, x:x+w]
    return cv2.resize(img, size)

# 1) PDF -> PNG (ทีละหน้า) ด้วย pypdfium2
pdf_to_images_pypdfium2(PDF_PATH, IMG_DIR, DPI)

# 2) ปรับให้ตรง + ขนาดเท่ากัน
for f in sorted(IMG_DIR.glob("*.png")):
    im = cv2.imread(str(f))
    im = deskew(im)
    im = crop_and_resize(im)
    cv2.imwrite(str(OUT_DIR / f.name), im)

print("✅ เสร็จสิ้น ->", OUT_DIR)
