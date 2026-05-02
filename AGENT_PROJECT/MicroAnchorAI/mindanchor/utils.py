import os
from typing import Dict, List

from PyPDF2 import PdfReader


def extract_pages_from_pdf(file_path: str) -> List[Dict[str, str]]:
	# Open the PDF file in binary mode
	with open(file_path, "rb") as file_handle:
		# Read the PDF with PyPDF2
		reader = PdfReader(file_handle)

		# Accumulate text for each page with its page number
		pages = []
		for index, page in enumerate(reader.pages, start=1):
			# Extract text from the page (may return None)
			page_text = page.extract_text() or ""
			pages.append({"page": index, "text": page_text})

	return pages


def split_text_into_chunks(
	text: str,
	chunk_size: int = 500,
	overlap: int = 50,
	max_chars: int = 200000,
	max_chunks: int = 2000,
) -> List[str]:
	# Limit very large pages to avoid memory spikes
	if len(text) > max_chars:
		text = text[:max_chars]

	# Normalize whitespace and trim extra spaces
	cleaned_text = " ".join(text.split())

	# Short-circuit if the text is empty
	if not cleaned_text:
		return []

	# Ensure overlap is smaller than chunk size
	if overlap >= chunk_size:
		overlap = max(0, chunk_size // 2)

	# Build overlapping chunks
	chunks = []
	start_index = 0
	text_length = len(cleaned_text)
	while start_index < text_length and len(chunks) < max_chunks:
		# Compute the chunk end index
		end_index = min(start_index + chunk_size, text_length)

		# Slice the chunk and store it
		chunk = cleaned_text[start_index:end_index]
		chunks.append(chunk)

		# Move the start index with overlap
		start_index = end_index - overlap
		if start_index < 0:
			start_index = 0

		# Prevent infinite loops when remaining text is smaller than overlap
		if start_index >= text_length:
			break

	return chunks


def split_pages_into_chunks(
	pages: List[Dict[str, str]],
	chunk_size: int = 500,
	overlap: int = 50,
	max_chars_per_page: int = 200000,
	max_chunks_per_page: int = 2000,
) -> List[Dict[str, str]]:
	# Split each page into chunks while preserving page numbers
	chunks = []
	for page in pages:
		page_number = page["page"]
		page_text = page["text"]
		page_chunks = split_text_into_chunks(
			page_text,
			chunk_size=chunk_size,
			overlap=overlap,
			max_chars=max_chars_per_page,
			max_chunks=max_chunks_per_page,
		)

		for chunk_text in page_chunks:
			chunks.append({"page": page_number, "text": chunk_text})

	return chunks


def is_pdf_file(filename: str) -> bool:
	# Validate the filename and extension
	if not filename:
		return False

	# Check the file extension
	_, ext = os.path.splitext(filename)
	return ext.lower() == ".pdf"
