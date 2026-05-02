# Data Flow

This document explains how data moves through MindAnchor AI.

## 1) PDF Upload and Chunking

- The user uploads a PDF in the UI.
- The server saves the file to `uploads/`.
- Each PDF page is extracted with PyPDF2.
- Each page is split into overlapping chunks.
- Each chunk keeps its page number for source tracking.

## 2) Embeddings and FAISS Index

- SentenceTransformers converts each chunk into an embedding vector.
- FAISS stores the vectors in an index on disk.
- The chunk metadata (text + page) is saved in `data/chunks.json`.

## 3) Question Answering (RAG)

- The user asks a question in the UI.
- The question is embedded and searched in FAISS.
- Top chunks are retrieved with similarity scores.
- The Groq LLM answers using the retrieved context.
- The UI shows the answer and the source chunks with page numbers.

## 4) Concept Extraction

- The system loads saved chunks.
- The Groq LLM extracts key concepts as a bullet list.
- The UI shows the concepts to the user.

## 5) Question Generation

- The extracted concepts are turned into recall questions.
- The UI shows the generated questions.

## 6) Spaced Repetition Scheduling

- Generated questions are stored in SQLite.
- Each question is scheduled with the 1-4-7 rule.
- The UI shows due questions for review.
- Reviewed items are recorded in history.

## Files and Storage

- `uploads/`: raw PDF uploads
- `data/faiss.index`: vector index
- `data/chunks.json`: chunk text + page numbers
- `database.db`: review schedule + history
