"""Parse scanned or digital question-paper PDFs with Groq-hosted vision models.

No Gemini dependency. Groq Vision and answer models share GROQ_API_KEY.
"""
import base64
import io
import json
import os
import re
import time

import fitz
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class VisionQuotaExceeded(RuntimeError):
    pass


def _parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("Vision model did not return valid JSON")


def _read_page(client, image_bytes, page_number):
    model = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    image_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
    prompt = f"""Read this EXAM QUESTION PAPER page {page_number} carefully.
Extract EVERY visible question, number, mark and OR choice. Preserve exact wording.
Do not answer questions. Never invent missing questions. Represent unclear text with
best reading and uncertain=true, and explain the issue in uncertainty_flags.
Return ONLY a JSON object with this shape:
{{"examination":{{}},"instructions":[],"sections":[{{"name":"PART A","questions":[
{{"number":"1","text":"question wording","marks":2,"source_page":{page_number},"uncertain":false}},
{{"number":"11","alternatives":[{{"label":"a","text":"option A","marks":13,"source_page":{page_number},"uncertain":false}},{{"label":"b","text":"option B","marks":13,"source_page":{page_number},"uncertain":false}}]}}
]}}],"uncertainty_flags":[{{"question_number":"13(b)","issue":"unclear text"}}]}}
Sections may be PART A/B/C etc. If a page has continuation of a question, capture the text
rather than dropping it. If no questions on a page, sections must be [].
"""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]}],
                temperature=0,
                max_tokens=5000,
                timeout=120,
            )
            return _parse_json(response.choices[0].message.content)
        except Exception as exc:
            code = getattr(exc, "status_code", None)
            msg = str(exc).lower()
            if code == 429 or "rate limit" in msg or "rate_limit" in msg or "quota" in msg:
                if "day" in msg or "daily" in msg:
                    raise VisionQuotaExceeded("Groq Vision daily quota reached. Retry after quota reset.") from exc
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise VisionQuotaExceeded("Groq Vision rate limit reached. Retry later.") from exc
            if code in (500, 502, 503, 504) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError("Groq Vision could not process this page")


def parse_question_paper_with_vision(pdf_bytes):
    """Compatible with existing main.py: returns parsed examination dictionary."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing from .env")
    client = Groq(api_key=api_key)
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    if not pdf.page_count:
        raise ValueError("Question paper is empty")
    max_pages = int(os.getenv("QUESTION_PAPER_MAX_PAGES", "10"))
    if pdf.page_count > max_pages:
        raise ValueError(f"Question paper exceeds {max_pages} pages. Increase QUESTION_PAPER_MAX_PAGES if needed.")

    result = {"examination": {}, "instructions": [], "sections": [], "uncertainty_flags": []}
    by_section = {}
    try:
        for i in range(pdf.page_count):
            page = pdf.load_page(i)
            # Reduce image dimensions to limit vision token use.
            pix = page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False)
            from PIL import Image
            im = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            if max(im.size) > 1700:
                im.thumbnail((1700, 1700))
            buffer = io.BytesIO()
            im.save(buffer, format="JPEG", quality=82)
            parsed = _read_page(client, buffer.getvalue(), i + 1)
            if not isinstance(parsed, dict):
                raise ValueError(f"Page {i + 1} produced an invalid response")
            if i == 0:
                result["examination"] = parsed.get("examination") or {}
            result["instructions"].extend(parsed.get("instructions") or [])
            result["uncertainty_flags"].extend(parsed.get("uncertainty_flags") or [])
            for section in parsed.get("sections") or []:
                name = str(section.get("name") or "Questions").strip()
                if name not in by_section:
                    by_section[name] = {"name": name, "questions": []}
                    result["sections"].append(by_section[name])
                by_section[name]["questions"].extend(section.get("questions") or [])
    finally:
        pdf.close()

    if not any(s["questions"] for s in result["sections"]):
        raise ValueError("No exam questions were detected in the PDF")
    return result
