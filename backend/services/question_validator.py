
from collections import Counter
from typing import Any


def _positive_number(value: Any):
    """Convert a mark value to a positive number."""
    try:
        if value is None or isinstance(value, bool):
            return None

        number = float(str(value).strip())

        if number <= 0 or number != number:
            return None

        return int(number) if number.is_integer() else number

    except (ValueError, TypeError, OverflowError):
        return None


def validate_question_paper(parsed_data: dict) -> dict:
    """
    Validate Gemini's question-paper JSON.

    Never invent missing questions, marks, or text.
    Never reject an entire paper for uncertain wording.
    """

    errors = []
    warnings = []
    review_items = []

    main_questions = 0
    answerable_questions = 0
    sections_checked = 0

    if not isinstance(parsed_data, dict):
        return {
            "valid": False,
            "errors": ["Parser output must be a JSON object."],
            "warnings": [],
            "review_items": [],
            "summary": {}
        }

    sections = parsed_data.get("sections")

    if not isinstance(sections, list) or not sections:
        errors.append("No valid sections found.")
        sections = []

    for section_index, section in enumerate(sections, start=1):

        if not isinstance(section, dict):
            errors.append(
                f"Section {section_index} is not an object."
            )
            continue

        sections_checked += 1

        section_name = (
            str(section.get("name") or "").strip()
            or f"Section {section_index}"
        )

        questions = section.get("questions")

        if not isinstance(questions, list):
            errors.append(
                f"{section_name}: questions must be a list."
            )
            continue

        seen_numbers = []

        for position, question in enumerate(
            questions, start=1
        ):
            if not isinstance(question, dict):
                errors.append(
                    f"{section_name}: question {position} "
                    "is not an object."
                )
                continue

            main_questions += 1

            number = str(
                question.get("number") or ""
            ).strip()

            identifier = (
                f"{section_name} Q{number}"
                if number
                else f"{section_name} position {position}"
            )

            if not number:
                warnings.append(
                    f"{identifier}: missing question number."
                )
            else:
                seen_numbers.append(number)

            alternatives = question.get("alternatives")

            if alternatives is not None:

                if (
                    not isinstance(alternatives, list)
                    or not alternatives
                ):
                    errors.append(
                        f"{identifier}: alternatives must "
                        "be a non-empty list."
                    )
                    continue

                labels = []

                for alternative_index, alternative in enumerate(
                    alternatives, start=1
                ):
                    if not isinstance(alternative, dict):
                        errors.append(
                            f"{identifier}: alternative "
                            f"{alternative_index} is invalid."
                        )
                        continue

                    label = str(
                        alternative.get("label") or ""
                    ).strip()

                    if label:
                        labels.append(label)

                    entry_id = (
                        f"{identifier} {label}"
                        if label
                        else f"{identifier} alternative "
                             f"{alternative_index}"
                    )

                    _validate_entry(
                        alternative,
                        entry_id,
                        errors,
                        warnings,
                        review_items
                    )

                    answerable_questions += 1

                duplicates = [
                    label
                    for label, count
                    in Counter(labels).items()
                    if count > 1
                ]

                if duplicates:
                    warnings.append(
                        f"{identifier}: duplicate alternative "
                        f"labels: {duplicates}"
                    )

            else:
                _validate_entry(
                    question,
                    identifier,
                    errors,
                    warnings,
                    review_items
                )

                answerable_questions += 1

        duplicates = [
            number
            for number, count
            in Counter(seen_numbers).items()
            if count > 1
        ]

        if duplicates:
            warnings.append(
                f"{section_name}: repeated question numbers "
                f"{duplicates}"
            )

    flags = parsed_data.get("uncertainty_flags") or []

    if isinstance(flags, list):
        for flag in flags:
            review_items.append({
                "source": "gemini",
                "detail": flag
            })
    else:
        warnings.append(
            "uncertainty_flags must be a list."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "review_items": review_items,
        "summary": {
            "sections": sections_checked,
            "main_questions": main_questions,
            "answerable_questions": answerable_questions,
            "review_items": len(review_items)
        }
    }


def _validate_entry(
    entry,
    identifier,
    errors,
    warnings,
    review_items
):
    """Validate a single question or OR alternative."""

    text = entry.get("text")

    if not isinstance(text, str) or not text.strip():
        errors.append(
            f"{identifier}: missing question text."
        )

    marks = _positive_number(entry.get("marks"))

    if marks is None:
        warnings.append(
            f"{identifier}: marks missing or invalid."
        )

    page = entry.get("source_page")

    if page is None or str(page).strip() == "":
        warnings.append(
            f"{identifier}: source page missing."
        )

    if entry.get("uncertain") is True:
        review_items.append({
            "source": "question",
            "question": identifier,
            "detail": "Gemini flagged uncertain wording."
        })
