import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# ---------------- QUIZ GENERATION PROMPT ----------------
quiz_template = """
Text:
{text}

You are an expert MCQ maker.

STRICT INSTRUCTIONS:
- Return ONLY valid JSON
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include text outside JSON

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

Create {number} multiple choice questions for {subject} students
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

    # Generate quiz
    quiz_response = quiz_chain.invoke(
        {
            "text": inputs["text"],
            "number": inputs["number"],
            "subject": inputs["subject"],
            "tone": inputs["tone"],
        }
    )

    # Parse once (only for review)
    quiz_data = json.loads(quiz_response)

    # Review quiz
    review_response = review_chain.invoke(
        {
            "subject": inputs["subject"],
            "quiz": quiz_data,
        }
    )

    return {
        "quiz": quiz_response,
        "review": review_response
    }
