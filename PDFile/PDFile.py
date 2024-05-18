from flask import Flask, request, render_template, redirect, url_for, session, current_app
import io
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load QA model and tokenizer
model_name = "distilbert/distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

with app.app_context():
    root_path = current_app.root_path
    file_path = os.path.join(root_path, "firebase.json")
    cred = credentials.Certificate(file_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

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
    return render_template('frontpage.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            users_ref = db.collection('PDFile')
            query = users_ref.where('username', '==', username).limit(1).get()  # Use get() instead of stream()
            user_doc = None
            for doc in query:
                user_doc = doc.to_dict()
                break

            if user_doc and user_doc['password'] == password:
                session['username'] = username
                return redirect(url_for('pdfile'))
            else:
                error = 'Invalid Credentials. Please try again.'
        except Exception as e:
            error = str(e)
    return render_template('login.html', error=error)

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/forgotpass')
def forgot_password():
    return render_template('forgotpass.html')

@app.route('/pdfile')
def pdfile():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('PDFile.html')

@app.route('/chat', methods=['POST'])
def chat():
    if 'username' not in session:
        return redirect(url_for('login'))

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
