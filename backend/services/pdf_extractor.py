
import io
import re
from functools import lru_cache

import pymupdf
import easyocr
import numpy as np
from PIL import Image


# Load EasyOCR only once
@lru_cache(maxsize=1)
def get_ocr_reader():
    """
    Initialize the EasyOCR model.
    Use CPU for text recognition.
    """
    return easyocr.Reader(["en"], gpu=False)


def extract_scanned_page(page):
    """
    Extract text from a scanned PDF page
    using EasyOCR at 300 DPI.
    """

    # Convert PDF page into an image
    pix = page.get_pixmap(
        dpi=300,
        alpha=False
    )

    # Convert image bytes to RGB
    image = Image.open(
        io.BytesIO(pix.tobytes("png"))
    ).convert("RGB")

    # Convert PIL image to NumPy array
    image_array = np.array(image)

    # Get cached OCR reader
    reader = get_ocr_reader()

    # Recognize text in the image
    results = reader.readtext(
        image_array,
        detail=1,
        paragraph=False
    )

    # Collect recognized text
    extracted_lines = []

    for bounding_box, text, confidence in results:
        extracted_lines.append(text)

    return "\n".join(extracted_lines)


def has_meaningful_text(text):
    """
    Determine whether a page contains
    sufficient embedded digital text.
    """

    cleaned = re.sub(r"\s+", "", text)

    alphanumeric_count = sum(
        character.isalnum()
        for character in cleaned
    )

    return alphanumeric_count >= 30


def extract_pdf_content(pdf_bytes, force_ocr=False):
    """
    Extract text from digital or scanned PDFs.

    force_ocr=True:
        Use EasyOCR on every page.

    force_ocr=False:
        Use PyMuPDF for digital pages.
        Use EasyOCR when digital text is insufficient.
    """

    if not pdf_bytes:
        raise ValueError("PDF file is empty.")

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    extracted_pages = []

    try:
        for page_number, page in enumerate(
            document,
            start=1
        ):

            digital_text = page.get_text("text")

            # Force OCR for scanned question papers
            if force_ocr:
                text = extract_scanned_page(page)
                method = "ocr"

            # Use digital text when available
            elif has_meaningful_text(digital_text):
                text = digital_text
                method = "digital"

            # Otherwise use OCR
            else:
                text = extract_scanned_page(page)
                method = "ocr"

            extracted_pages.append({
                "page": page_number,
                "text": text.strip(),
                "extraction_method": method
            })

    finally:
        document.close()

    return {
        "total_pages": len(extracted_pages),
        "pages": extracted_pages,
        "ocr_pages": sum(
            p["extraction_method"] == "ocr"
            for p in extracted_pages
        ),
        "digital_pages": sum(
            p["extraction_method"] == "digital"
            for p in extracted_pages
        )
    }
