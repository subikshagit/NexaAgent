# MindAnchor AI

MindAnchor AI is a simple RAG + spaced repetition demo built with Flask, SQLite, FAISS, and SentenceTransformers.

## Setup

1) Create a virtual environment (recommended):

```bash
python -m venv .venv
```

2) Activate it:

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

3) Install dependencies:

```bash
pip install -r requirements.txt
```

4) Create a .env file for your Gemini key:

PowerShell:

```powershell
Copy .env.example to .env and update the value:

```bash
cp .env.example .env
```

Then set:

```
GEMINI_API_KEY=YOUR_KEY_HERE
```
```

## Run

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Quick Flow

1) Upload a PDF
2) Ask a question
3) Extract concepts and generate questions
4) Schedule reviews and mark them reviewed
