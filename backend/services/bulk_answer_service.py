"""Flatten exam questions and connect FAISS retrieval to Groq answers."""
from backend.rag.vector_store import search_vector_index
from backend.rag.groq_answer_service import generate_exam_answer


def flatten_questions(parsed_data):
    result = []
    for section in parsed_data.get("sections", []):
        name = section.get("name", "Section")
        for question in section.get("questions", []):
            number = str(question.get("number") or "")
            alternatives = question.get("alternatives")
            if isinstance(alternatives, list) and alternatives:
                for alternative in alternatives:
                    label = str(alternative.get("label") or "")
                    result.append({
                        "id": number + label, "section": name, "number": number,
                        "alternative": label, "question": alternative.get("text", ""),
                        "marks": alternative.get("marks"),
                        "source_page": alternative.get("source_page"),
                        "uncertain": alternative.get("uncertain", False),
                    })
            else:
                result.append({
                    "id": number, "section": name, "number": number,
                    "alternative": None, "question": question.get("text", ""),
                    "marks": question.get("marks"),
                    "source_page": question.get("source_page"),
                    "uncertain": question.get("uncertain", False),
                })
    return result


def create_rag_answer_function(index, chunks):
    def answer_question(question_text, marks):
        matches = search_vector_index(index, chunks, question_text, top_k=8)
        return generate_exam_answer(question_text, matches, marks, allow_general=True)
    return answer_question


def generate_bulk_answers(parsed_data, answer_function):
    questions = flatten_questions(parsed_data)
    results = []
    for q in questions:
        try:
            response = answer_function(q["question"], q["marks"])
            if not response.get("answer") or response.get("source") == "insufficient":
                raise RuntimeError("No usable answer")
            results.append({**q, "status": "completed", **response})
        except Exception as exc:
            results.append({**q, "status": "failed", "answer": None,
                            "source": None, "pages": [], "error": str(exc)})
    failed = sum(q["status"] == "failed" for q in results)
    return {"total_questions": len(results), "completed": len(results) - failed,
            "failed": failed, "results": results}
