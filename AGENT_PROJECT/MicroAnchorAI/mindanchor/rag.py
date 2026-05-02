import json
import logging
import os
import time
from typing import Dict, List, Tuple

import faiss
import httpx
import numpy as np
from google import genai

from sentence_transformers import SentenceTransformer

# Define model and storage paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
FALLBACK_MODELS = os.getenv(
	"GEMINI_FALLBACK_MODELS",
	"models/gemini-2.0-flash,models/gemini-flash-latest,models/gemini-flash-lite-latest",
)


# Keep global objects to avoid reloading
_embedder = None
_index_cache = None
_chunks_cache = None
_gemini_client = None
_last_gemini_error = None


def _set_last_gemini_error(message: str | None) -> None:
	# Track the last Gemini API error for UI feedback
	global _last_gemini_error
	_last_gemini_error = message


def get_last_gemini_error() -> str | None:
	# Return the most recent Gemini API error message
	return _last_gemini_error

def get_embedder() -> SentenceTransformer:
	# Reuse the model if already loaded
	global _embedder
	if _embedder is None:
		# Load the sentence-transformer model
		_embedder = SentenceTransformer(MODEL_NAME)
	return _embedder


def get_gemini_client() -> genai.Client | None:
	# Reuse the Gemini client if already initialized
	global _gemini_client
	if _gemini_client is not None:
		return _gemini_client

	# Read the API key from the environment
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		return None

	_gemini_client = genai.Client(api_key=api_key)
	return _gemini_client


def _friendly_gemini_error_message(exc: Exception) -> str:
	# Map common connection errors to a user-friendly message
	if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
		retry_delay = _extract_retry_delay_seconds(str(exc))
		if retry_delay is not None:
			return (
				"Gemini API quota exceeded (429). "
				f"Retry in about {retry_delay:.0f}s, or use a different model/plan."
			)
		return (
			"Gemini API quota exceeded (429). "
			"Wait a bit or use a different model or billing plan."
		)
	if "503" in str(exc) or "UNAVAILABLE" in str(exc):
		return "Gemini API is temporarily overloaded (503). Try again shortly."
	if isinstance(exc, httpx.ConnectError) or "getaddrinfo failed" in str(exc):
		return (
			"Unable to reach the Gemini API. "
			"Check your internet connection, DNS, or proxy settings and try again."
		)
	return f"Gemini API error: {exc}"


def _is_retryable_gemini_error(exc: Exception) -> bool:
	# Retry on temporary load or network failures
	message = str(exc)
	return (
		"429" in message
		or "RESOURCE_EXHAUSTED" in message
		or "503" in message
		or "UNAVAILABLE" in message
		or isinstance(exc, httpx.ConnectError)
	)


def _extract_retry_delay_seconds(message: str) -> float | None:
	# Parse retry delay hints from Gemini errors
	if "retry in" in message:
		parts = message.split("retry in", 1)[1]
		value = "".join(ch for ch in parts if ch.isdigit() or ch == ".")
		if value:
			try:
				return float(value)
			except ValueError:
				return None
	if "retryDelay" in message and "s" in message:
		value = "".join(ch for ch in message if ch.isdigit() or ch == ".")
		if value:
			try:
				return float(value)
			except ValueError:
				return None
	return None


def _parse_fallback_models() -> List[str]:
	# Normalize and filter fallback models from environment
	models = []
	for item in (FALLBACK_MODELS or "").split(","):
		model_name = item.strip()
		if model_name:
			models.append(normalize_model_name(model_name))
	return models


def _iter_candidate_models(primary_model: str) -> List[str]:
	# Build the ordered list of models to try
	primary = normalize_model_name(primary_model)
	fall_back = [model for model in _parse_fallback_models() if model != primary]
	return [primary] + fall_back


def _call_gemini(client: genai.Client, model: str, prompt: str):
	# Centralized Gemini call with retries and fallback models
	last_exc = None
	for candidate_model in _iter_candidate_models(model):
		for attempt in range(3):
			try:
				response = client.models.generate_content(
					model=candidate_model,
					contents=prompt,
				)
				_set_last_gemini_error(None)
				return response
			except Exception as exc:
				last_exc = exc
				message = str(exc)
				if "404" in message or "NOT_FOUND" in message:
					logging.warning("Gemini model not available: %s", candidate_model)
					break
				if _is_retryable_gemini_error(exc) and attempt < 2:
					retry_delay = _extract_retry_delay_seconds(str(exc))
					if retry_delay is not None:
						time.sleep(min(retry_delay, 10))
					else:
						time.sleep(0.5 * (2 ** attempt))
					continue
				if _is_retryable_gemini_error(exc):
					break
				logging.exception("Gemini API call failed")
				_set_last_gemini_error(_friendly_gemini_error_message(exc))
				return None

	if last_exc is not None:
		logging.exception("Gemini API call failed after fallbacks", exc_info=last_exc)
		_set_last_gemini_error(_friendly_gemini_error_message(last_exc))
	return None


def normalize_model_name(model_name: str) -> str:
	# Ensure the model name includes the required prefix
	if model_name.startswith("models/"):
		return model_name
	return f"models/{model_name}"


def list_supported_models() -> List[str]:
	# Return the list of model names available to this API key
	client = get_gemini_client()
	if client is None:
		return []

	try:
		models = client.models.list()
		return [model.name for model in models]
	except Exception:
		_set_last_gemini_error("Unable to list Gemini models. Check your API key and network.")
		return []


def build_embeddings(chunks: List[Dict[str, str]] | List[str]) -> np.ndarray:
	# Load the model
	embedder = get_embedder()

	# Extract text content from chunks
	if chunks and isinstance(chunks[0], str):
		chunk_texts = [str(item) for item in chunks]
	else:
		chunk_texts = [item["text"] for item in chunks]

	# Create embeddings for each chunk
	embeddings = embedder.encode(
		chunk_texts,
		convert_to_numpy=True,
		normalize_embeddings=True,
	)

	return embeddings.astype("float32")


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
	# Get the vector dimension
	dimension = embeddings.shape[1]

	# Create an inner-product index (works with normalized vectors)
	index = faiss.IndexFlatIP(dimension)

	# Add embeddings to the index
	index.add(embeddings)

	return index


def save_index(index: faiss.IndexFlatIP) -> None:
	# Ensure the data directory exists
	os.makedirs(DATA_DIR, exist_ok=True)

	# Persist the FAISS index to disk
	faiss.write_index(index, INDEX_PATH)


def save_chunks(chunks: List[Dict[str, str]]) -> None:
	# Ensure the data directory exists
	os.makedirs(DATA_DIR, exist_ok=True)

	# Save chunks for later retrieval
	with open(CHUNKS_PATH, "w", encoding="utf-8") as file_handle:
		json.dump(chunks, file_handle, indent=2)


def build_and_save_index(chunks: List[Dict[str, str]]) -> Tuple[int, int]:
	# Short-circuit if no chunks
	if not chunks:
		return 0, 0

	# Build embeddings for the chunks
	embeddings = build_embeddings(chunks)

	# Create the FAISS index
	index = create_faiss_index(embeddings)

	# Save index and chunks to disk
	save_index(index)
	save_chunks(chunks)

	# Return counts for reporting
	return len(chunks), embeddings.shape[1]


def load_index() -> faiss.IndexFlatIP | None:
	# Return cached index if available
	global _index_cache
	if _index_cache is not None:
		return _index_cache

	# Check if the index file exists
	if not os.path.exists(INDEX_PATH):
		return None

	# Load the FAISS index from disk
	_index_cache = faiss.read_index(INDEX_PATH)
	return _index_cache


def load_chunks() -> List[Dict[str, str]]:
	# Return cached chunks if available
	global _chunks_cache
	if _chunks_cache is not None:
		return _chunks_cache

	# Check if the chunks file exists
	if not os.path.exists(CHUNKS_PATH):
		return []

	# Load chunks from disk
	with open(CHUNKS_PATH, "r", encoding="utf-8") as file_handle:
		_chunks_cache = json.load(file_handle)

	return _chunks_cache


def retrieve_chunks_with_scores(question: str, top_k: int = 3) -> List[dict]:
	# Load index and chunks
	index = load_index()
	chunks = load_chunks()

	# Stop if index or chunks are missing
	if index is None or not chunks:
		return []

	# Build embedding for the query
	query_embedding = build_embeddings([question])

	# Search for the most similar chunks
	scores, indices = index.search(query_embedding, top_k)

	# Map indices to chunk text and similarity score
	retrieved = []
	for rank, idx in enumerate(indices[0]):
		if 0 <= idx < len(chunks):
			retrieved.append(
				{
					"chunk_id": int(idx),
					"text": chunks[idx]["text"],
					"page": chunks[idx]["page"],
					"score": float(scores[0][rank]),
				}
			)

	return retrieved


def build_prompt(question: str, context_text: str) -> str:
	# Create a helpful, grounded prompt for the LLM
	return (
		"You are MindAnchor AI, a study assistant. Use the provided context first. "
		"If the context is thin or missing key details, say it is a best-effort answer "
		"and ask one clarifying question or suggest uploading more material. "
		"Answer in 3 to 6 sentences, then add 2 to 4 bullet takeaways if useful.\n\n"
		f"Context:\n{context_text}\n\n"
		f"Question: {question}\n"
		"Answer:"
	)


def truncate_text(text: str, max_chars: int = 240) -> str:
	# Short-circuit if the text already fits
	if len(text) <= max_chars:
		return text

	# Trim and add an ellipsis
	return text[: max_chars - 3].rstrip() + "..."


def answer_question(question: str, top_k: int = 3) -> dict:
	# Retrieve relevant chunks for the query
	retrieved = retrieve_chunks_with_scores(question, top_k=top_k)

	# Handle missing context
	if not retrieved:
		return {
			"answer": "No indexed knowledge found. Please upload a PDF first.",
			"sources": [],
		}

	# Build a limited context from top chunks
	context_text = build_context_from_chunks([item["text"] for item in retrieved])

	# Build the LLM prompt
	prompt = build_prompt(question, context_text)

	# Initialize the Gemini client
	client = get_gemini_client()
	if client is None:
		return {
			"answer": "GEMINI_API_KEY is not set. Please update your .env file.",
			"sources": [],
		}

	# Prepare sources for UI display
	sources = [
		{
			"chunk_id": item["chunk_id"],
			"text": truncate_text(item["text"]),
			"score": item["score"],
			"page": item["page"],
		}
		for item in retrieved
	]

	# Call the LLM to generate an answer
	response = _call_gemini(client, normalize_model_name(LLM_MODEL), prompt)
	if response is None:
		return {
			"answer": get_last_gemini_error() or "Unable to reach the Gemini API.",
			"sources": sources,
		}

	# Extract the response text
	return {
		"answer": (response.text or "").strip(),
		"sources": sources,
	}


def build_context_from_chunks(chunks: List[str], max_chars: int = 4000) -> str:
	# Combine chunks until the size limit is reached
	collected = []
	current_size = 0
	for chunk in chunks:
		# Stop if adding this chunk exceeds the max size
		if current_size + len(chunk) > max_chars:
			break

		# Store the chunk and update the size
		collected.append(chunk)
		current_size += len(chunk)

	# Join chunks into a single context block
	return "\n\n".join(collected)


def parse_list_items(raw_text: str) -> List[str]:
	# Split the text into lines
	lines = raw_text.splitlines()

	# Normalize each line and remove bullets
	items = []
	for line in lines:
		clean = line.strip()
		if clean.startswith("- "):
			clean = clean[2:]
		if clean.startswith("* "):
			clean = clean[2:]
		if clean:
			items.append(clean)

	return items


def extract_concepts(max_items: int = 8) -> List[str]:
	# Load chunks from disk
	chunks = load_chunks()

	# Stop if no chunks are available
	if not chunks:
		return []

	# Build a limited context for the LLM
	chunk_texts = [item["text"] for item in chunks]
	context_text = build_context_from_chunks(chunk_texts)

	# Prepare the extraction prompt
	prompt = (
		"Extract the most important concepts as short sentences. "
		"Return a bullet list with one concept per line. "
		f"Limit to {max_items} items.\n\n"
		f"Text:\n{context_text}"
	)

	# Initialize the Gemini client
	client = get_gemini_client()
	if client is None:
		return []

	# Request concept extraction
	response = _call_gemini(client, normalize_model_name(LLM_MODEL), prompt)
	if response is None:
		return []

	# Parse and return the concepts
	return parse_list_items(response.text or "")[:max_items]


def generate_questions(max_items: int = 8) -> List[str]:
	# Extract concepts to build questions from
	concepts = extract_concepts(max_items=max_items)

	# Stop if no concepts are available
	if not concepts:
		return []

	# Build a simple prompt from concepts
	concept_text = "\n".join(f"- {item}" for item in concepts)
	prompt = (
		"Create simple recall questions for each concept. "
		"Return a bullet list with one question per line.\n\n"
		f"Concepts:\n{concept_text}"
	)

	# Initialize the Gemini client
	client = get_gemini_client()
	if client is None:
		return []

	# Request question generation
	response = _call_gemini(client, normalize_model_name(LLM_MODEL), prompt)
	if response is None:
		return []

	# Parse and return the questions
	return parse_list_items(response.text or "")[:max_items]
