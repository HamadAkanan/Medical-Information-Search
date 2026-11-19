from collections import defaultdict
import pandas as pd
from preprocessing import preprocess_text


def build_inverted_index(
    df: pd.DataFrame,
    remove_stopwords: bool = True,
) -> tuple[dict[str, set[str]], set[str], dict[str, str]]:
    """
    Construct a term -> set(doc_id) inverted index using Python dicts and sets.
    Returns:
        inverted_index: dict mapping each processed term to a set of doc_ids
        all_doc_ids: complete set of indexed doc_ids (for NOT complement operations)
        doc_titles: dict mapping doc_id -> title for readable output
    """
    inverted_index: dict[str, set[str]] = defaultdict(set)
    all_doc_ids: set[str] = set()
    doc_titles: dict[str, str] = {}

    for _, row in df.iterrows():
        doc_id = str(row["doc_id"])
        title = str(row["title"])
        text = str(row["text"])

        all_doc_ids.add(doc_id)
        doc_titles[doc_id] = title

        terms = preprocess_text(text, remove_stopwords=remove_stopwords)
        for term in set(terms):
            inverted_index[term].add(doc_id)

    return dict(inverted_index), all_doc_ids, doc_titles
