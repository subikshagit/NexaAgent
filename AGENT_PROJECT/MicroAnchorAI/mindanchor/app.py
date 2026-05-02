import os
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from dotenv import load_dotenv

from rag import (
    answer_question,
    build_and_save_index,
    extract_concepts,
    generate_questions,
    get_last_gemini_error,
    list_supported_models,
)
from scheduler import add_review_items, get_due_items, get_review_history, init_db, mark_reviewed
from utils import extract_pages_from_pdf, is_pdf_file, split_pages_into_chunks

# Create the Flask app
app = Flask(__name__)

# Load environment variables from .env if present
load_dotenv()

# Configure upload folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize the SQLite database
init_db()


@app.route("/", methods=["GET"])
def index():
    # Render the simple chat interface
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    # Get the uploaded file
    uploaded_file = request.files.get("pdf_file")

    # Handle missing file
    if not uploaded_file or uploaded_file.filename == "":
        return render_template("index.html", upload_error="Please select a PDF file.")

    # Validate file extension
    if not is_pdf_file(uploaded_file.filename):
        return render_template("index.html", upload_error="Only PDF files are allowed.")

    # Save the file securely
    safe_name = secure_filename(uploaded_file.filename)
    saved_path = os.path.join(UPLOAD_DIR, safe_name)
    uploaded_file.save(saved_path)

    # Extract page text from the PDF
    pages = extract_pages_from_pdf(saved_path)

    # Split the pages into chunks while preserving page numbers
    chunks = split_pages_into_chunks(pages)

    # Stop if chunking failed or returned nothing
    if not chunks:
        return render_template(
            "index.html",
            upload_error="No text could be extracted from the PDF.",
        )

    # Build embeddings and save the FAISS index
    chunk_count, embedding_dim = build_and_save_index(chunks)

    # Return a confirmation message
    return render_template(
        "index.html",
        upload_success="PDF uploaded and processed.",
        uploaded_filename=safe_name,
        chunk_count=chunk_count,
        embedding_dim=embedding_dim,
    )


@app.route("/ask", methods=["POST"])
def ask():
    # Read the user question from the form
    question = request.form.get("question", "").strip()

    # Handle empty questions
    if not question:
        return render_template(
            "index.html",
            question="",
            answer="Please enter a question.",
            sources=[],
        )

    # Use RAG to answer the question
    result = answer_question(question)
    answer = result["answer"]
    sources = result["sources"]

    # Render the page with the response
    return render_template(
        "index.html",
        question=question,
        answer=answer,
        sources=sources,
    )


@app.route("/concepts", methods=["POST"])
def concepts():
    # Ensure the Gemini key is available
    if not os.getenv("GEMINI_API_KEY"):
        return render_template(
            "index.html",
            concepts_message="GEMINI_API_KEY is not set. Update your .env file.",
        )

    # Extract key concepts from the indexed chunks
    concepts_list = extract_concepts()

    # Handle missing concepts
    if not concepts_list:
        error_message = get_last_gemini_error()
        if error_message:
            return render_template(
                "index.html",
                concepts_message=error_message,
            )
        return render_template(
            "index.html",
            concepts_message="No concepts found. Upload a PDF first.",
        )

    # Render the concepts on the page
    return render_template("index.html", concepts=concepts_list)


@app.route("/questions", methods=["POST"])
def questions():
    # Ensure the Gemini key is available
    if not os.getenv("GEMINI_API_KEY"):
        return render_template(
            "index.html",
            questions_message="GEMINI_API_KEY is not set. Update your .env file.",
        )

    # Generate recall questions from extracted concepts
    questions_list = generate_questions()

    # Handle missing questions
    if not questions_list:
        error_message = get_last_gemini_error()
        if error_message:
            return render_template(
                "index.html",
                questions_message=error_message,
            )
        return render_template(
            "index.html",
            questions_message="No questions generated. Upload a PDF first.",
        )

    # Render the questions on the page
    return render_template("index.html", questions=questions_list)


@app.route("/schedule", methods=["POST"])
def schedule_reviews():
    # Ensure the Gemini key is available
    if not os.getenv("GEMINI_API_KEY"):
        return render_template(
            "index.html",
            schedule_message="GEMINI_API_KEY is not set. Update your .env file.",
        )

    # Generate recall questions to schedule
    questions_list = generate_questions()

    # Handle missing questions
    if not questions_list:
        error_message = get_last_gemini_error()
        if error_message:
            return render_template(
                "index.html",
                schedule_message=error_message,
            )
        return render_template(
            "index.html",
            schedule_message="No questions generated. Upload a PDF first.",
        )

    # Add the questions to the review schedule
    scheduled_count = add_review_items(questions_list)

    # Load due items for display
    due_items = get_due_items()

    # Render confirmation and due items
    return render_template(
        "index.html",
        schedule_message=f"Scheduled {scheduled_count} review items.",
        due_items=due_items,
    )


@app.route("/reviews", methods=["GET"])
def reviews():
    # Load due items
    due_items = get_due_items()

    # Load review history
    history_items = get_review_history()

    # Render the due list
    return render_template("index.html", due_items=due_items, history_items=history_items)


@app.route("/models", methods=["GET"])
def models():
    # Return available Gemini models as JSON
    return jsonify({"models": list_supported_models()})


@app.route("/review/<int:item_id>", methods=["POST"])
def review_item(item_id: int):
    # Mark the review item as completed
    mark_reviewed(item_id)

    # Reload due items
    due_items = get_due_items()

    # Reload review history
    history_items = get_review_history()

    # Render the updated list
    return render_template(
        "index.html",
        due_items=due_items,
        history_items=history_items,
        review_message="Review recorded. Next date scheduled.",
    )


if __name__ == "__main__":
    # Run the Flask development server
    app.run(debug=True)
