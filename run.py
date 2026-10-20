"""Run frontend and backend on http://127.0.0.1:8000."""
import uvicorn

if __name__ == "__main__":
    print("StudyLens AI: http://127.0.0.1:8000")
    print("API documentation: http://127.0.0.1:8000/docs")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
