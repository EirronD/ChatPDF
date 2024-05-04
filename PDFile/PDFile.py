from flask import Flask, request, render_template
import io
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

app = Flask(__name__)

# Load QA model and tokenizer
model_name = "distilbert/distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_content):
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise Exception("Error extracting text from PDF.")

# Function to answer the question
def answer_question(question, context):
    try:
        answer = qa_pipeline(question=question, context=context)
        answer_text = answer["answer"]
        return answer_text
    except Exception as e:
        raise Exception("Error answering the question.")

@app.route('/')
def index():
    return render_template('PDFile.html')

@app.route('/chat', methods=['POST'])
def chat():
    pdf_file = request.files['pdf_file']
    pdf_content = pdf_file.read()
    
    if not pdf_content:
        return "Please upload a PDF file."
    
    user_question = request.form['user_query']

    if not user_question:
        return "Please provide a question."

    try:
        pdf_text = extract_text_from_pdf(pdf_content)
        answer = answer_question(user_question, pdf_text)
        return f"{answer}"
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
