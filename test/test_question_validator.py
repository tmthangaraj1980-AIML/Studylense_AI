
import json
from pathlib import Path

from backend.services.question_validator import (
    validate_question_paper
)


input_path = Path("vision_parsed_questions.json")

parsed_data = json.loads(
    input_path.read_text(encoding="utf-8")
)

report = validate_question_paper(parsed_data)

print("\nQUESTION VALIDATION REPORT")
print("=" * 50)

print(json.dumps(
    report,
    indent=4,
    ensure_ascii=False
))

Path("validation_report.json").write_text(
    json.dumps(
        report,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print("\nSaved: validation_report.json")
