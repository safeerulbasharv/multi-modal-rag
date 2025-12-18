# ocr_loader.py

import fitz
import pytesseract
from PIL import Image
from langchain.schema import Document
import io

def load_ocr_from_pdf(path, min_text_len=200):
    """
    OCR ONLY pages with little selectable text.
    Fast + safe.
    """
    docs = []
    pdf = fitz.open(path)

    for page_num, page in enumerate(pdf):
        text = page.get_text().strip()

        if len(text) > min_text_len:
            continue

        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        ocr_text = pytesseract.image_to_string(image)

        if ocr_text.strip():
            docs.append(
                Document(
                    page_content=ocr_text,
                    metadata={
                        "source": path,
                        "page": page_num,
                        "modality": "ocr"
                    }
                )
            )

    return docs
