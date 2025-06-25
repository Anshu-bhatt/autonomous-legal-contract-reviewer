import pandas as pd
import re
import spacy
import os
import csv

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Custom clause cleaner function
def clean_clause(text):
    if pd.isnull(text):
        return ""

    text = text.lower().strip()

    # Remove fill-in-the-blank patterns: dashes, underscores, dots (2 or more)
    text = re.sub(r'[_\-–—\.]{2,}', ' ', text)

    # Remove all punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Lemmatize and remove stop words
    doc = nlp(text)
    lemmatized = " ".join([
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha
    ])

    return lemmatized

# === MAIN EXECUTION ===

# File paths
input_path = "outputs/extracted_clauses.csv"
output_path = "outputs/cleaned_clauses.csv"

# Load clause data
df = pd.read_csv(input_path)

# Clean clauses
cleaned_clauses = df['clause_text'].apply(clean_clause)

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# Save as quoted, comma-separated values (one per line)
with open(output_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    for clause in cleaned_clauses:
        writer.writerow([clause])  # single column row, quoted
