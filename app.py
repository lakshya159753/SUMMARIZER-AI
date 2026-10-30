import os
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types, errors
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load .env if it exists
load_dotenv()

app = Flask(__name__)

# --- Config -----------------------------------------------------------------
# Get a free key at https://aistudio.google.com/apikey
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set. Create a free key at "
        "https://aistudio.google.com/apikey and add it to your .env file."
    )

# Change the model in .env (GEMINI_MODEL=...) without touching the code.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Safety cap so a huge PDF doesn't blow past the free-tier token limits.
MAX_INPUT_CHARS = 200_000

SYSTEM_PROMPT = (
    "You are a helpful study assistant that summarizes text into clear bullet points. "
    "Use plain text only: start each bullet with '- ' and do not use markdown "
    "formatting such as ** or #."
)

client = genai.Client(api_key=api_key)


def summarize_text(text):
    """Uses Google's Gemini API (free tier) to summarize text into key points."""
    config = {
        "system_instruction": SYSTEM_PROMPT,
        "temperature": 0.4,
        # Gemini 2.5 models "think" first and thinking tokens count toward this
        # limit, so leave plenty of headroom for the actual summary.
        "max_output_tokens": 2048,
    }
    # Thinking isn't needed for summaries; turning it off is faster and
    # saves free-tier quota. (Only 2.5 Flash models allow budget 0.)
    if MODEL.startswith("gemini-2.5-flash"):
        config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    response = client.models.generate_content(
        model=MODEL,
        contents=f"Summarize this text:\n\n{text[:MAX_INPUT_CHARS]}",
        config=types.GenerateContentConfig(**config),
    )

    if not response.text:
        raise ValueError(
            "Gemini returned no text (the content may have been blocked). "
            "Try different input."
        )
    return response.text.strip()


def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file"""
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        text = ""

        # Text input
        if 'text' in request.form and request.form['text'].strip():
            text = request.form['text']

        # File upload
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file)
            elif file.filename.lower().endswith('.txt'):
                text = file.read().decode('utf-8', errors='replace')

        # Validate text
        if not text or len(text.strip()) < 20:
            return jsonify({"error": "Input is too short. Please provide at least 20 characters."}), 400

        # Summarize text
        summary = summarize_text(text)
        return jsonify({"summary": summary})

    except errors.APIError as e:
        if e.code == 429:
            return jsonify({
                "error": "Free-tier rate limit reached. Wait a minute and try again."
            }), 429
        return jsonify({"error": f"Gemini API error ({e.code}): {e.message}"}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Bound to localhost: debug mode on 0.0.0.0 would expose the debugger to your network.
    app.run(host='127.0.0.1', port=5000, debug=True)
