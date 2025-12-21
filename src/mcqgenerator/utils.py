import os
import json
import traceback
import PyPDF2


def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text

        except Exception:
            raise Exception("Error reading the PDF file")

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    else:
        raise Exception("Unsupported file format. Only PDF and TXT are supported.")


def get_table_data(quiz_input):
    """
    Accepts either:
    - JSON string returned by LLM, OR
    - Parsed Python dict

    Returns:
    - List of rows suitable for table / DataFrame
    """

    try:
        # Step 1: Ensure we have a dict
        if isinstance(quiz_input, str):
            quiz_data = json.loads(quiz_input)
        elif isinstance(quiz_input, dict):
            quiz_data = quiz_input
        else:
            raise TypeError("quiz_input must be str or dict")

        quiz_table_data = []

        # Step 2: quiz is a LIST under key "quiz"
        for q in quiz_data["quiz"]:
            mcq = q["question"]

            options = " || ".join(
                f"{opt} -> {text}" for opt, text in q["options"].items()
            )

            correct = q["answer"]

            quiz_table_data.append({
                "MCQ": mcq,
                "Choices": options,
                "Correct": correct
            })

        return quiz_table_data

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return False
