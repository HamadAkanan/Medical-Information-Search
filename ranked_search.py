import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_to_string


class TFIDFRankedRetriever:
    """
    Vector Space Model retriever using TF-IDF weighting and cosine similarity.
    Fits the vectorizer once on the corpus and transforms incoming queries without refitting.
    """

    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = None
        self.df = None

    def fit(self, df: pd.DataFrame) -> None:
        """Fit TF-IDF vectorizer on preprocessed corpus documents."""
        self.df = df.reset_index(drop=True).copy()
        processed_corpus = [
            preprocess_to_string(text, remove_stopwords=self.remove_stopwords)
            for text in self.df["text"]
        ]
        self.doc_vectors = self.vectorizer.fit_transform(processed_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Transform query using the already-fitted vectorizer, compute cosine similarity,
        and return the top-k highest-scoring documents.
        """
        if self.doc_vectors is None or self.df is None:
            raise RuntimeError("Retriever must be fitted with .fit(df) before searching.")

        if not isinstance(query, str) or not query.strip():
            return []

        processed_query = preprocess_to_string(query, remove_stopwords=self.remove_stopwords)
        if not processed_query:
            return []

        # Transform query using existing vocabulary & IDF weights (never fit_transform)
        query_vector = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vector, self.doc_vectors).flatten()

        # Sort descending by cosine similarity score
        ranked_indices = scores.argsort()[::-1]

        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0.0:
                break
            results.append({
                "rank": len(results) + 1,
                "doc_id": self.df.loc[idx, "doc_id"],
                "title": self.df.loc[idx, "title"],
                "category": self.df.loc[idx, "category"],
                "score": round(score, 4),
            })
            if len(results) >= top_k:
                break

        return results
