from flask import Flask, request, render_template
import io
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key'

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

def find_supporting_evidence(answer, context):
    pattern = rf"\b{re.escape(answer)}\b"
    sentences = re.findall(r"[^\.!\?]+[\.!\?]", context, flags=re.DOTALL)
    for sentence in sentences:
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            cleaned_sentence = re.sub(r'\d+', '', sentence)
            cleaned_sentence = re.sub(r'[^\w\s]', '', cleaned_sentence)
            cleaned_sentence = " ".join(cleaned_sentence.split())
            return cleaned_sentence.strip()  # Return only the relevant part of the sentence
    return ""  # Return an empty string if no evidence found

def get_page_numbers(evidence_sentence, context, window_size=50):
    pattern = r"(?:Page|p\.)\s*(\d+)"
    matches = re.findall(pattern, context, flags=re.IGNORECASE)
    page_numbers = []

    if not matches:
        return "not available"

    sentence_start = context.find(evidence_sentence)

    for match in matches:
        page_num = int(match)
        if sentence_start - window_size < context.find(match) < sentence_start + window_size:
            page_numbers.append(str(page_num))

    if page_numbers:
        return ", ".join(page_numbers)
    else:
        return "not available"

def re_search_pdf(question, context):
    try:
        paragraphs = context.split("\n\n")
        best_answer = None
        best_supporting_evidence = None

        for paragraph in paragraphs:
            answer = qa_pipeline(question=question, context=paragraph)

            if not best_answer or answer["score"] > best_answer["score"]:
                best_answer = answer
                best_supporting_evidence = find_supporting_evidence(answer["answer"], paragraph)

        if best_answer:
            formatted_answer = best_answer["answer"]
            page_numbers = get_page_numbers(best_supporting_evidence, context)
            return {
                "formatted_answer": formatted_answer,
                "supporting_evidence": best_supporting_evidence,
                "page_numbers": page_numbers
            }
        else:
            return None

    except Exception as e:
        raise Exception(f"Error re-searching the PDF: {e}")

def answer_question(question, context):
    try:
        answer_info = re_search_pdf(question, context)

        if answer_info:
            formatted_answer = answer_info["formatted_answer"]
            supporting_evidence = answer_info["supporting_evidence"]
            page_numbers = answer_info["page_numbers"]
            full_answer = f"The answer to your question '{question}' is: {formatted_answer}. "
            full_answer += f"According to the PDF, {supporting_evidence}. "
            full_answer += f"This information is found on page(s): {page_numbers}."
        else:
            answer = qa_pipeline(question=question, context=context)
            answer_text = answer["answer"]
            evidence = find_supporting_evidence(answer_text, context)
            full_answer = f"The answer to your question '{question}' is: {answer_text}. "
            if evidence:
                full_answer += f"According to the PDF, {evidence}. "
            else:
                full_answer += "No additional context found in the PDF."

        return full_answer

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
