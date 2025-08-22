# -*- coding: utf-8 -*-
"""
PDF -> ภาพ -> Align ด้วย Template (ORB+Homography) -> Resize A4 -> รวม PDF
- ถ้ามี TEMPLATE_PATH จะใช้เป็นต้นแบบ
- ถ้าไม่ระบุ จะใช้หน้าแรกของ PDF เป็นต้นแบบให้หน้าอื่น ๆ
"""
from pathlib import Path
import cv2, numpy as np
import pypdfium2 as pdfium
from PIL import Image

# ======= ตั้งค่า =======
PDF_PATH       = Path(r"D:\A\Workbook Code\Repo\OCR-Project\Input\Work.pdf")  # ไฟล์ PDF ต้นทาง
TEMPLATE_PATH  = Path(r"D:\A\Workbook Code\Repo\OCR-Project\Template.jpg")    # ไฟล์รูปต้นแบบ (PNG/JPG) ถ้าไม่มีให้ใส่ None
OUT_IMG_DIR    = Path(r"D:\A\Workbook Code\Repo\OCR-Project\aligned_pages")   # โฟลเดอร์ภาพหลังจัดแนว
OUT_PDF        = Path(r"D:\A\Workbook Code\Repo\OCR-Project\Output\Work01.pdf")
DPI_RENDER     = 400
A4_PX          = (2480, 3508)  # (กว้าง, สูง) A4 @~300dpi

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# ======= ฟังก์ชันหลักที่จำเป็น =======
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

def align_to_template(page_bgr, template_bgr,
                      nfeatures=3000, good_ratio=0.75, ransac_thresh=5.0):
    """Align page -> template ด้วย ORB + BF + Lowe Ratio + findHomography(RANSAC)"""
    # ทำขาวดำเพื่อความเสถียร
    img1 = ensure_gray(template_bgr)
    img2 = ensure_gray(page_bgr)

    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=7)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)

    if d1 is None or d2 is None or len(k1)<10 or len(k2)<10:
        return None  # ให้ผู้เรียกตัดสินใจ fallback

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d1, d2, k=2)

    good = []
    for m,n in knn:
        if m.distance < good_ratio * n.distance:
            good.append(m)

    if len(good) < 15:  # น้อยไปจะไม่เสถียร
        return None

    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)
    if H is None:
        return None

    h, w = template_bgr.shape[:2]
    warped = cv2.warpPerspective(page_bgr, H, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped

def binarized_page_rect(img_bgr):
    """หา bounding rect ของเนื้อกระดาษเพื่อครอปขอบรบกวนออก (อย่างง่าย)"""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0,0,img_bgr.shape[1], img_bgr.shape[0]
    x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return x,y,w,h

def resize_a4(img_bgr, a4_size=A4_PX):
    return cv2.resize(img_bgr, a4_size, interpolation=cv2.INTER_CUBIC)

def images_to_pdf(image_paths, out_pdf: Path):
    if not image_paths:
        raise ValueError("ไม่มีภาพให้รวม PDF")
    pages = [Image.open(p).convert("RGB") for p in image_paths]
    pages[0].save(out_pdf, save_all=True, append_images=pages[1:], resolution=300)

# ======= ทำงานจริง =======
if __name__ == "__main__":
    assert PDF_PATH.exists(), f"ไม่พบไฟล์: {PDF_PATH}"

    # 1) แปลง PDF เป็นภาพ
    pages = pdf_to_images(PDF_PATH, DPI_RENDER)

    # 2) เตรียม Template
    if TEMPLATE_PATH and TEMPLATE_PATH.exists():
        template = cv2.imread(str(TEMPLATE_PATH))
    else:
        # ใช้หน้าแรกเป็นต้นแบบให้หน้าที่เหลือ
        template = pages[0].copy()

    # 3) จัดแนวทุกหน้าให้ตรงกับ Template
    out_paths = []
    for i, page in enumerate(pages, start=1):
        aligned = align_to_template(page, template)

        if aligned is None:
            # Fallback ง่าย ๆ: ถ้า Align ไม่ได้ ให้หมุนเป็น Portrait และครอปขอบ
            h,w = page.shape[:2]
            if w > h:
                page = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
            x,y,ww,hh = binarized_page_rect(page)
            aligned = page[y:y+hh, x:x+ww]

        # ครอปเล็กน้อยอีกครั้ง (กันเศษขอบหลัง warp) + Resize A4
        x,y,w,h = binarized_page_rect(aligned)
        aligned = aligned[y:y+h, x:x+w]
        aligned = resize_a4(aligned, A4_PX)

        out_p = OUT_IMG_DIR / f"page_{i:03}.png"
        cv2.imwrite(str(out_p), aligned)
        out_paths.append(out_p)
        print(f"[OK] page {i:03} -> {out_p.name}")

    # 4) รวมกลับเป็น PDF เดียว
    images_to_pdf(out_paths, OUT_PDF)
    print("✅ เสร็จสิ้น →", OUT_PDF)
