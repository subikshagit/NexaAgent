import os
import sqlite3
from datetime import date, timedelta
from typing import List, Tuple

# Define database location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")

# Define the spaced repetition intervals (1-4-7 rule)
REVIEW_INTERVALS = [1, 4, 7]


def init_db() -> None:
	# Connect to the SQLite database
	connection = sqlite3.connect(DB_PATH)

	# Create a cursor for executing SQL
	cursor = connection.cursor()

	# Create the reviews table if it does not exist
	cursor.execute(
		"""
		CREATE TABLE IF NOT EXISTS reviews (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			question TEXT NOT NULL,
			stage INTEGER NOT NULL,
			next_review TEXT NOT NULL
		)
		"""
	)

	# Create a review history table for audit tracking
	cursor.execute(
		"""
		CREATE TABLE IF NOT EXISTS review_history (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			review_id INTEGER NOT NULL,
			reviewed_at TEXT NOT NULL,
			stage INTEGER NOT NULL,
			FOREIGN KEY (review_id) REFERENCES reviews (id)
		)
		"""
	)

	# Commit changes and close the connection
	connection.commit()
	connection.close()


def add_review_items(questions: List[str]) -> int:
	# Stop if there is nothing to schedule
	if not questions:
		return 0

	# Compute the first review date (today + 1 day)
	first_review = (date.today() + timedelta(days=REVIEW_INTERVALS[0])).isoformat()

	# Connect to the SQLite database
	connection = sqlite3.connect(DB_PATH)
	cursor = connection.cursor()

	# Insert each question as a new review item
	for question in questions:
		cursor.execute(
			"INSERT INTO reviews (question, stage, next_review) VALUES (?, ?, ?)",
			(question, 0, first_review),
		)

	# Commit changes and close the connection
	connection.commit()
	connection.close()

	# Return the number of items scheduled
	return len(questions)


def get_due_items(on_date: date | None = None) -> List[Tuple[int, str, str]]:
	# Use today's date if no date is provided
	target_date = on_date or date.today()
	target_date_str = target_date.isoformat()

	# Connect to the SQLite database
	connection = sqlite3.connect(DB_PATH)
	cursor = connection.cursor()

	# Select all items due on or before the target date
	cursor.execute(
		"SELECT id, question, next_review FROM reviews WHERE next_review <= ?",
		(target_date_str,),
	)

	# Fetch results and close the connection
	rows = cursor.fetchall()
	connection.close()

	return rows


def mark_reviewed(item_id: int) -> None:
	# Connect to the SQLite database
	connection = sqlite3.connect(DB_PATH)
	cursor = connection.cursor()

	# Fetch the current stage for this item
	cursor.execute("SELECT stage FROM reviews WHERE id = ?", (item_id,))
	row = cursor.fetchone()

	# Stop if the item does not exist
	if not row:
		connection.close()
		return

	# Compute the next stage
	current_stage = row[0]
	next_stage = min(current_stage + 1, len(REVIEW_INTERVALS) - 1)

	# Compute the next review date
	interval_days = REVIEW_INTERVALS[next_stage]
	next_review = (date.today() + timedelta(days=interval_days)).isoformat()

	# Update the item with the new stage and review date
	cursor.execute(
		"UPDATE reviews SET stage = ?, next_review = ? WHERE id = ?",
		(next_stage, next_review, item_id),
	)

	# Insert a history record for this review
	reviewed_at = date.today().isoformat()
	cursor.execute(
		"INSERT INTO review_history (review_id, reviewed_at, stage) VALUES (?, ?, ?)",
		(item_id, reviewed_at, next_stage),
	)

	# Commit changes and close the connection
	connection.commit()
	connection.close()


def get_review_history(limit: int = 20) -> List[Tuple[int, int, str, int]]:
	# Connect to the SQLite database
	connection = sqlite3.connect(DB_PATH)
	cursor = connection.cursor()

	# Fetch the most recent review events
	cursor.execute(
		"""
		SELECT id, review_id, reviewed_at, stage
		FROM review_history
		ORDER BY id DESC
		LIMIT ?
		""",
		(limit,),
	)

	rows = cursor.fetchall()
	connection.close()

	return rows
