import json
import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------- LLM (OLLAMA – NO KEY) ----------------
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0.6
)

# ---------------- HELPER: SAFE JSON EXTRACTION ----------------
def extract_json(text: str) -> str:
    """
    Extract the first valid JSON object from LLM output.
    Required because Ollama may add extra text.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in model output")
    return match.group()

# ---------------- QUIZ GENERATION PROMPT ----------------
quiz_template = """
Text:
{text}

You are an expert MCQ maker.

STRICT INSTRUCTIONS (VERY IMPORTANT):
- Return ONLY valid JSON
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include text outside JSON
- JSON must be parseable using json.loads()

The JSON format MUST be exactly:

{{
  "quiz": [
    {{
      "question": "string",
      "options": {{
        "A": "string",
        "B": "string",
        "C": "string",
        "D": "string"
      }},
      "answer": "A"
    }}
  ]
}}

Create exactly {number} multiple choice questions for {subject} students
in a {tone} tone. Ensure questions are unique and based only on the text.
"""

quiz_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone"],
    template=quiz_template
)

quiz_chain = quiz_prompt | llm | StrOutputParser()

# ---------------- QUIZ REVIEW PROMPT ----------------
review_template = """
You are an expert educator and English writer.

Review the following quiz for {subject} students.
Analyze difficulty (max 50 words) and suggest improvements if needed.

Quiz:
{quiz}
"""

review_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=review_template
)

review_chain = review_prompt | llm | StrOutputParser()

# ---------------- PUBLIC FUNCTION ----------------
def generate_evaluate_chain(inputs: dict) -> dict:
    """
    Inputs:
    {
        "text": str,
        "number": int,
        "subject": str,
        "tone": str
    }

    Returns:
    {
        "quiz": JSON string,
        "review": str
    }
    """

    # -------- Generate quiz (raw text) --------
    raw_quiz_output = quiz_chain.invoke(
        {
            "text": inputs["text"],
            "number": inputs["number"],
            "subject": inputs["subject"],
            "tone": inputs["tone"],
        }
    )

    # -------- Extract & parse JSON safely --------
    quiz_json_str = extract_json(raw_quiz_output)
    quiz_data = json.loads(quiz_json_str)

    # -------- Review quiz --------
    review_response = review_chain.invoke(
        {
            "subject": inputs["subject"],
            "quiz": quiz_data,
        }
    )

    return {
        "quiz": quiz_json_str,   # clean JSON string
        "review": review_response
    }
