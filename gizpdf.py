import streamlit as st
import json
import io
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from PyPDF2 import PdfReader

model_name = "deepset/bert-base-cased-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def main():
    st.title("PDF Chatbot")

    pdf_file = st.file_uploader("Upload your PDF", type='pdf')

    if pdf_file is not None:
        pdf_content = pdf_file.read()
        pdf_text = extract_text_from_pdf(pdf_content)

        user_question = st.text_input("Ask a question:", "")

        if user_question:
            answer = answer_question(user_question, pdf_text)

            st.write("Bot:")
            st.write("- " + answer)

            feedback = st.radio("Was this answer helpful?", ["Yes", "No"])

            if feedback == "No":
                correct_answer = st.text_input("Please provide the correct answer:", "")
                if correct_answer:
                    add_to_dataset(user_question, correct_answer)

def extract_text_from_pdf(pdf_content):
    pdf_file = io.BytesIO(pdf_content)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def answer_question(question, context):
    answer = qa_pipeline(question=question, context=context)
    return answer["answer"]

def add_to_dataset(question, answer):
    dataset_file = "dataset.json"
    try:
        with open(dataset_file, "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        dataset = []

    dataset.append({"question": question, "answer": answer})

    with open(dataset_file, "w") as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    main()
