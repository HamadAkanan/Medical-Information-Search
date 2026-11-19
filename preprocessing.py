import re
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Try NLTK English stopwords first; fall back cleanly to scikit-learn's English stopword list
try:
    from nltk.corpus import stopwords
    _BASE_STOPWORDS = set(stopwords.words("english"))
except LookupError:
    _BASE_STOPWORDS = set(ENGLISH_STOP_WORDS)

_STEMMER = PorterStemmer()


def tokenize(text: str) -> list[str]:
    """Lowercase text and extract alphabetic word tokens."""
    if not isinstance(text, str) or not text.strip():
        return []
    return re.findall(r"\b[a-z]+\b", text.lower())


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    custom_stopwords: set[str] | None = None,
    apply_stemming: bool = True,
) -> list[str]:
    """
    Standardized preprocessing pipeline applied to both documents and user queries:
    1. Lowercasing & regex tokenization
    2. Configurable stopword filtering
    3. Porter stemming
    """
    tokens = tokenize(text)
    if not tokens:
        return []

    active_stopwords = set(_BASE_STOPWORDS)
    if custom_stopwords:
        active_stopwords.update(w.lower() for w in custom_stopwords)

    processed_terms = []
    for token in tokens:
        if remove_stopwords and token in active_stopwords:
            continue
        term = _STEMMER.stem(token) if apply_stemming else token
        processed_terms.append(term)

    return processed_terms


def preprocess_to_string(
    text: str,
    remove_stopwords: bool = True,
    custom_stopwords: set[str] | None = None,
    apply_stemming: bool = True,
) -> str:
    """Join preprocessed tokens into a single string for TF-IDF vectorization."""
    return " ".join(
        preprocess_text(
            text,
            remove_stopwords=remove_stopwords,
            custom_stopwords=custom_stopwords,
            apply_stemming=apply_stemming,
        )
    )


def demonstrate_preprocessing(csv_path: str = "dataset/documents.csv", doc_id: str = "D36") -> None:
    """Display original text, initial tokens, and final processed terms for a sample record."""
    df = pd.read_csv(csv_path)
    match = df[df["doc_id"] == doc_id]
    record = match.iloc[0] if not match.empty else df.iloc[0]

    raw_text = record["text"]
    initial_tokens = tokenize(raw_text)
    final_terms = preprocess_text(raw_text, remove_stopwords=True, apply_stemming=True)

    print("=== STEP 3: PREPROCESSING BEFORE / AFTER EXAMPLE ===")
    print(f"Document ID : {record['doc_id']} (Category: {record['category']})")
    print(f"Title       : {record['title']}\n")
    print(f"1. Original Text (snippet):\n{raw_text[:280]}...\n")
    print(f"2. Initial Tokens (first 20):\n{initial_tokens[:20]}\n")
    print(f"3. Final Processed Terms (first 20):\n{final_terms[:20]}\n")
    print(f"Summary: Reduced {len(initial_tokens)} raw tokens -> {len(final_terms)} indexed terms.")


if __name__ == "__main__":
    demonstrate_preprocessing()
