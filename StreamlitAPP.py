import json
import traceback
import pandas as pd
import streamlit as st

from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.MCQGen import generate_evaluate_chain

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="MCQ Generator (Ollama)", layout="wide")
st.title("🧠📘 MCQ Generator & Evaluator using LLM (Ollama)")

with st.form("user_inputs"):
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"]
    )

    mcq_count = st.number_input(
        "Number of Questions",
        min_value=1,
        max_value=20,
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

            # Step 2: Call Ollama chain (NO CALLBACK)
            result = generate_evaluate_chain(
                {
                    "text": text,
                    "number": mcq_count,
                    "subject": subject,
                    "tone": tone,
                }
            )

            # Step 3: Convert quiz JSON → table
            # result["quiz"] is already a clean JSON string
            quiz_table_data = get_table_data(result["quiz"])

            if not quiz_table_data:
                st.error("Failed to generate quiz table.")
                st.stop()

            quiz_df = pd.DataFrame(quiz_table_data)
            quiz_df.index += 1

            # ---------------- DISPLAY ----------------
            st.subheader("Generated MCQs")
            st.table(quiz_df)

            st.subheader("Quiz Review")
            st.text_area(
                label="Review",
                value=result["review"],
                height=150
            )

            # ---------------- INFO ----------------
            with st.expander("Model Info"):
                st.write("Model: **gemma3:4b (Ollama – Local)**")
                st.write("No API key • No billing • Offline inference")

        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            st.error(f"Error: {e}")
