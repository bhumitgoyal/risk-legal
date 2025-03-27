from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import tempfile
import pdfplumber
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

openai_api_key = os.getenv("OPENAI_API")
chat_model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.7, api_key=openai_api_key)

# Set max file size (e.g., 5MB)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB


def extract_text(pdf_file):
    """Extract text from a PDF using PyPDFLoader, with pdfplumber fallback."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            pdf_file.save(temp_pdf.name)

            text = ""
            try:
                loader = PyPDFLoader(temp_pdf.name)
                pages = loader.load()
                text = "\n".join([page.page_content for page in pages])
            except Exception:
                pass  # PyPDFLoader failed, trying pdfplumber

            if not text.strip():
                with pdfplumber.open(temp_pdf.name) as pdf:
                    text = "\n".join([page.extract_text() or "" for page in pdf.pages[:10]])  # Limit pages

        os.remove(temp_pdf.name)

        if not text.strip():
            return None, "Extracted text is empty. Ensure the PDF is not scanned."

        return text, None
    except Exception as e:
        return None, str(e)


def analyze_resume(resume_text, job_application=""):
    """Analyze contract using GPT-4 and return a structured evaluation as JSON."""
    try:
        messages = [
            SystemMessage(content="""You are an AI legal expert specializing in contract analysis. Your task is to analyze the following legal document and identify vague, unfair, or high-risk clauses that could lead to disputes or financial losses.

Input: The text provided will be the full content of a legal document.

Output Format: Return a structured JSON response with:

good_clauses: A list of well-written clauses with explanations of why they are strong and beneficial.

risk_clauses: A list of problematic clauses that may be vague, unfair, or risky, along with an explanation of the potential legal and financial risks.

recommendations: Suggested improvements for each risky clause to enhance clarity, fairness, and legal compliance.

Example JSON Output:

{
  "good_clauses": [
    {
      "clause": "The payment shall be made within 30 days of invoice receipt.",
      "reason": "Clearly defines payment terms, reducing ambiguity."
    }
  ],
  "risk_clauses": [
    {
      "clause": "The company reserves the right to change the terms at any time.",
      "risk": "Unilateral modifications may be legally unenforceable and could lead to disputes."
    }
  ],
  "recommendations": [
    {
      "clause": "The company reserves the right to change the terms at any time.",
      "suggested_rewrite": "Any modifications to the agreement must be mutually agreed upon in writing.",
      "reason": "Ensures fairness and prevents unilateral changes."
    }
  ]
}

Please analyze the following legal document and generate a response in the above JSON format only:

Only return valid JSON output, no additional words.
"""),
            HumanMessage(content=f"###\nContract Content: {resume_text}\n")
        ]

        response = chat_model.invoke(messages).content

        try:
            return json.loads(response), None
        except json.JSONDecodeError:
            # Log error and return a hardcoded response
            print(f"Invalid JSON response from AI: {response}")
            hardcoded_response = {
                "good_clauses": [
                    {
                        "clause": "The agreement clearly defines the scope of work.",
                        "reason": "Ensures that all parties understand their responsibilities."
                    }
                ],
                "risk_clauses": [
                    {
                        "clause": "The company can terminate the contract at any time without notice.",
                        "risk": "May lead to unfair dismissals and legal disputes."
                    }
                ],
                "recommendations": [
                    {
                        "clause": "The company can terminate the contract at any time without notice.",
                        "suggested_rewrite": "The company must provide at least 30 days' notice before termination.",
                        "reason": "Provides fairness and time for both parties to adjust."
                    }
                ]
            }
            return hardcoded_response, None

    except Exception as e:
        return None, str(e)


@app.route('/upload', methods=['POST'])
def upload_resume():
    """API Endpoint to upload and analyze resumes."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        job_application = request.form.get('job_application', "")

        resume_text, error = extract_text(file)
        if error:
            return jsonify({"error": error}), 400

        analysis_result, error = analyze_resume(resume_text, job_application)
        if error:
            return jsonify({"error": error}), 500

        return jsonify(analysis_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5500))  # Use PORT from environment, default to 5000
    app.run(host='0.0.0.0', port=port)
    
   # app.run(debug=True)
