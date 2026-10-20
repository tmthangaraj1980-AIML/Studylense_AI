
import json
from pathlib import Path

from backend.services.bulk_answer_service import (
    flatten_questions,
    generate_bulk_answers
)


data = json.loads(
    Path("vision_parsed_questions.json").read_text(
        encoding="utf-8"
    )
)

questions = flatten_questions(data)

print("\nEXTRACTED ANSWERABLE QUESTIONS")
print("=" * 50)

for question in questions:
    print(
        f"Q{question['id']} | "
        f"{question['marks']} marks | "
        f"{question['section']}"
    )

print("\nTotal:", len(questions))


def mock_answer(question_text, marks):
    """
    Test the pipeline without spending API tokens.
    """

    return {
        "answer": (
            f"TEST ANSWER for: {question_text}"
        ),
        "source": "test",
        "pages": []
    }


result = generate_bulk_answers(
    data,
    mock_answer
)

assert result["total_questions"] == len(questions)
assert result["completed"] == len(questions)
assert result["failed"] == 0

Path("bulk_test_results.json").write_text(
    json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print("\nBULK PIPELINE TEST")
print("=" * 50)

print("Total:", result["total_questions"])
print("Completed:", result["completed"])
print("Failed:", result["failed"])
print("Saved: bulk_test_results.json")
