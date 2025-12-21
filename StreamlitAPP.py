import json
import traceback
import pandas as pd
from dotenv import load_dotenv

import streamlit as st

from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.MCQGen import generate_evaluate_chain
from langchain_community.callbacks import get_openai_callback

# Load environment variables
load_dotenv()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="MCQ Generator", layout="wide")
st.title("MCQ Generator and Evaluator")

with st.form("user_inputs"):
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"]
    )

    mcq_count = st.number_input(
        "Number of Questions",
        min_value=1,
        max_value=50,
        value=5
    )

    subject = st.text_input("Subject", max_chars=30)

    tone = st.text_input(
        "Complexity level of questions",
        max_chars=20,
        placeholder="easy / medium / hard"
    )

    submit_button = st.form_submit_button("Create MCQs")

# ---------------- MAIN LOGIC ----------------
if submit_button and uploaded_file and subject and tone:
    with st.spinner("Generating MCQs..."):
        try:
            # Step 1: Read file
            text = read_file(uploaded_file)

            # Step 2: Call LLM
            with get_openai_callback() as cb:
                result = generate_evaluate_chain(
                    {
                        "text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone,
                    }
                )

            # Step 3: Parse quiz JSON
            quiz_data = json.loads(result["quiz"])
            review_text = result["review"]

            # Step 4: Convert to table
            quiz_table_data = get_table_data(quiz_data)

            if not quiz_table_data:
                st.error("Failed to generate quiz table.")
                st.stop()

            quiz_df = pd.DataFrame(quiz_table_data)
            quiz_df.index += 1

            # ---------------- DISPLAY ----------------
            st.subheader("Generated MCQs")
            st.table(quiz_df)

            st.subheader("Quiz Review")
            st.text_area("Review", value=review_text, height=150)

            # ---------------- TOKEN USAGE ----------------
            with st.expander("Token Usage"):
                st.write(f"Total Tokens: {cb.total_tokens}")
                st.write(f"Prompt Tokens: {cb.prompt_tokens}")
                st.write(f"Completion Tokens: {cb.completion_tokens}")
                st.write(f"Total Cost ($): {cb.total_cost}")

        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            st.error(f"Error: {e}")
