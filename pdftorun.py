# -*- coding: utf-8 -*-
# จัดแนวเอกสารทั้งโฟลเดอร์: PDF -> Align ตาม Template (หรือหน้าแรก) -> Resize A4 -> รวมกลับเป็น PDF
from pathlib import Path
import cv2, numpy as np
import pypdfium2 as pdfium
from PIL import Image

# ======= ตั้งค่า =======
INPUT_DIR     = Path(r"D:\A\Workbook Code\Repo\OCR-Project\Input")     # โฟลเดอร์ PDF ต้นทาง
OUT_ROOT      = Path(r"D:\A\Workbook Code\Repo\OCR-Project\Output")    # โฟลเดอร์ผลลัพธ์
TEMPLATE_PATH = Path(r"D:\A\Workbook Code\Repo\OCR-Project\Template.jpg") # ตัวเลือก: Path(r"D:\A\Workbook Code\Repo\OCR-Project\template.png") หรือ None
DPI_RENDER    = 400
A4_PX         = (2480, 3508)  # (กว้าง,สูง) A4 ~300dpi

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ======= ฟังก์ชันที่จำเป็น =======
def pil_to_cv2(pil):
    arr = np.array(pil)
    return arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def pdf_to_images(pdf_path: Path, dpi=400):
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi/72.0
    pages = []
    for i in range(len(doc)):
        pil = doc[i].render(scale=scale).to_pil()
        pages.append(pil_to_cv2(pil))
    return pages

def ensure_gray(img):
    return img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def align_to_template(page_bgr, template_bgr, nfeatures=3000, good_ratio=0.75, ransac_thresh=5.0):
    # ORB + BF + Lowe Ratio + Homography(RANSAC)
    img1 = ensure_gray(template_bgr)
    img2 = ensure_gray(page_bgr)
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=7)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None or len(k1)<10 or len(k2)<10:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d1, d2, k=2)
    good = [m for m,n in knn if m.distance < good_ratio*n.distance]
    if len(good) < 15:
        return None
    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)  # template
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)  # page
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)
    if H is None:
        return None
    h, w = template_bgr.shape[:2]
    return cv2.warpPerspective(page_bgr, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def find_page_rect(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0,0,img_bgr.shape[1], img_bgr.shape[0]
    x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return x,y,w,h

def to_portrait(img_bgr):
    h,w = img_bgr.shape[:2]
    return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE) if w>h else img_bgr

def resize_a4(img_bgr, a4=A4_PX):
    return cv2.resize(img_bgr, a4, interpolation=cv2.INTER_CUBIC)

def images_to_pdf(image_paths, out_pdf: Path):
    pages = [Image.open(p).convert("RGB") for p in image_paths]
    pages[0].save(out_pdf, save_all=True, append_images=pages[1:], resolution=300)

def process_pdf(pdf_path: Path, template_global_bgr=None):
    # เตรียมโฟลเดอร์ย่อย
    name = pdf_path.stem
    out_dir = OUT_ROOT / name
    img_dir = out_dir / "aligned_pages"
    out_pdf = out_dir / f"{name}_aligned.pdf"
    img_dir.mkdir(parents=True, exist_ok=True)

    pages = pdf_to_images(pdf_path, DPI_RENDER)
    if not pages:
        return

    # เลือก Template: global > หน้าแรกของไฟล์นี้
    if template_global_bgr is not None:
        template = template_global_bgr
    else:
        template = pages[0].copy()

    out_paths = []
    for i, page in enumerate(pages, start=1):
        aligned = align_to_template(page, template)
        if aligned is None:
            # Fallback: บังคับเป็นแนวตั้ง + ครอปขอบ
            page = to_portrait(page)
            x,y,w,h = find_page_rect(page)
            aligned = page[y:y+h, x:x+w]
        # ครอปกันขอบซ้ำ + รีไซส์ A4
        x,y,w,h = find_page_rect(aligned)
        aligned = aligned[y:y+h, x:x+w]
        aligned = resize_a4(aligned, A4_PX)

        out_p = img_dir / f"page_{i:03}.png"
        cv2.imwrite(str(out_p), aligned)
        out_paths.append(out_p)

    images_to_pdf(out_paths, out_pdf)
    print(f"[OK] {pdf_path.name} -> {out_pdf}")

# ======= รันทั้งโฟลเดอร์ =======
if __name__ == "__main__":
    # โหลด Template กลางถ้ามี
    tpl_bgr = cv2.imread(str(TEMPLATE_PATH)) if TEMPLATE_PATH else None

    pdfs = sorted([p for p in INPUT_DIR.glob("*.pdf") if p.is_file()])
    if not pdfs:
        raise FileNotFoundError(f"ไม่พบไฟล์ PDF ในโฟลเดอร์: {INPUT_DIR}")

    for pdf in pdfs:
        process_pdf(pdf, template_global_bgr=tpl_bgr)

    print("✅ เสร็จสิ้นทั้งหมด")
