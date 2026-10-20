# StudyLens AI — RAG Examination Answer Generator

A full-stack AI portfolio project that converts a textbook PDF and an examination question-paper PDF into an organized, downloadable **PDF answer bank**.

## Features

- Upload and index a textbook PDF.
- Extract questions from digital/scanned exam papers with a Groq-hosted vision model, including OR alternatives.
- Retrieve relevant textbook passages using Sentence Transformers and FAISS.
- Generate mark-aware answers with Groq-hosted GPT-OSS 20B.
- Label answers grounded in textbook passages versus general AI knowledge.
- Display progress and retry failed answers.
- Export the answer bank as **PDF**; ask questions about the textbook.
- Single FastAPI server hosts the REST API and HTML/CSS/JavaScript frontend.

## Tech stack

Python, FastAPI, Uvicorn, PyMuPDF, Sentence Transformers, FAISS, GroqCloud API, ReportLab, HTML, CSS, JavaScript. Some PDFs may require an OCR fallback (EasyOCR, if installed/configured).

## Architecture

```text
Textbook PDF → text extraction / OCR → chunks → embeddings → FAISS
Question PDF → page images → Groq Vision → parsed questions
Parsed question → FAISS retrieval → Groq text LLM → answer + source label
Answer bank → PDF export
Frontend → FastAPI REST endpoints
```

## Local installation (Windows)

1. Clone the repository and open its root folder.
2. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env`. Set `GROQ_API_KEY` and `GROQ_VISION_MODEL` to the **working** vision model available to your account. Never commit `.env`.
4. Start the app:

   ```powershell
   python run.py
   ```
5. Visit `http://127.0.0.1:8000/`.

The existing project `requirements.txt`, `run.py`, `backend/` and other project files are required; this update ZIP deliberately changes only the frontend and adds deployment/docs files.

## Usage

1. Upload textbook PDF and wait for indexing.
2. Upload question-paper PDF and review extracted questions.
3. Generate answers; retry failures if quota permits.
4. Review source labels and download the PDF answer bank.

## Demonstration results (one test, not an accuracy benchmark)

- Textbook: **170 pages**, indexed into **381 chunks**.
- Question paper: **22 answerable entries**, including alternatives.
- **21/22 answers generated** in the most recent UI demonstration.
- These counts do **not** establish answer correctness. Verify extracted wording, citations and generated answers independently.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Health check |
| POST | `/api/sessions/upload` | Index textbook PDF |
| POST | `/api/sessions/{id}/questions/upload` | Extract questions |
| GET | `/api/sessions/{id}/questions` | Review extracted questions |
| POST | `/api/sessions/{id}/answers/generate` | Generate answer bank |
| GET | `/api/sessions/{id}/answers/progress` | Poll progress |
| GET | `/api/sessions/{id}/answers` | Fetch results |
| POST | `/api/sessions/{id}/answers/retry` | Retry failed answers |
| GET | `/api/sessions/{id}/answers/export/pdf` | Download PDF |
| POST | `/api/sessions/{id}/ask` | Ask a textbook question |
| DELETE | `/api/sessions/{id}` | Remove session |

The backend may still expose a legacy DOCX endpoint, but the portfolio UI intentionally offers PDF only.

## Deploy a public live demo with Hugging Face Spaces (Docker)

> Hosting quotas and available free hardware can change. A public demo may sleep when idle and wake slowly. This app downloads an embedding model and may use OCR; **check RAM, disk and startup logs** before treating deployment as production-ready.

1. Push the **full working project** to GitHub (not only the files in this update). Keep `.env` out of Git.
2. Ensure the repository root contains `Dockerfile`, `requirements.txt`, `backend/`, `web/` and `run.py`.
3. At [Hugging Face Spaces](https://huggingface.co/new-space), create a **Docker** Space, choosing a free CPU option if offered.
4. Push the full project into the Space repository. For Docker Spaces, include this metadata at the top of the Space repository's `README.md` (above the project README content):

   ```yaml
   ---
   title: StudyLens AI
   emoji: 📚
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   ---
   ```
5. Under **Space Settings → Variables and secrets**, add `GROQ_API_KEY` as a **Secret**, and add `GROQ_MODEL` and `GROQ_VISION_MODEL` as variables with model IDs known to work on your account.
6. Wait for the Docker build; check logs. Open the Space URL and first verify `/api/health` returns `{"status":"ok"}`.
7. Test a **small textbook and short question paper first**. Monitor Groq rate limits and memory usage. If OCR or embedding model loading exceeds free CPU limits, simplify OCR, use smaller PDFs or select appropriate hardware.
8. Add the public Space URL to your GitHub README and portfolio once the live test succeeds.

**Important:** The current server stores sessions and FAISS indexes in process memory. They disappear on restart/sleep and are not shared between multiple workers. This is acceptable for a small demonstration, not persistent production hosting. Uploaded files and question content are processed by third-party Groq APIs; avoid private/sensitive documents. The app is not a grading or verified-answer system.

## Deployment checklist

- [ ] `.env` and API keys excluded from Git
- [ ] `requirements.txt` includes all actually imported packages
- [ ] Groq vision model ID verified with your account
- [ ] `/api/health` works publicly
- [ ] Upload → extraction → generation → PDF download tested publicly
- [ ] Public demo URL and screenshots added to portfolio

## Future improvements

- Validate answer accuracy against a manually reviewed question/answer set.
- Add persistent session storage and per-user quotas.
- Improve equation formatting and mathematical symbol rendering in PDF.
- Add authentication and upload limits appropriate for public access.

## License

Add a license if you intend to distribute or accept contributions.
