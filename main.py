import os
import pandas as pd
from prepare_dataset import build_medquad_dataset
from preprocessing import demonstrate_preprocessing
from inverted_index import build_inverted_index
from boolean_search import evaluate_boolean_query
from ranked_search import TFIDFRankedRetriever


def sort_doc_ids(doc_ids: set[str]) -> list[str]:
    """Sort IDs like 'D1', 'D2', 'D10' numerically."""
    return sorted(doc_ids, key=lambda d: int(d[1:]) if d[1:].isdigit() else d)


def run_demo_suite(
    df: pd.DataFrame,
    inverted_index: dict[str, set[str]],
    all_doc_ids: set[str],
    doc_titles: dict[str, str],
    ranker: TFIDFRankedRetriever,
) -> None:
    """Run all required Phase 1 demonstrations (Steps 3, 4, and 5)."""
    print("\n" + "=" * 65)
    demonstrate_preprocessing("dataset/documents.csv", doc_id="D36")

    print("\n" + "=" * 65)
    print("=== STEP 4: INVERTED INDEX & BOOLEAN SEARCH TESTS ===")
    for sample_term in ["diabet", "symptom", "treatment"]:
        postings = sort_doc_ids(inverted_index.get(sample_term, set()))
        print(f"Postings['{sample_term}'] ({len(postings)} docs): {postings[:8]}")

    boolean_queries = [
        "diabetes AND treatment",
        "diabetes OR insulin",
        "diabetes AND NOT gestational",
        "NOT diabetes",
    ]
    for bq in boolean_queries:
        matches = sort_doc_ids(evaluate_boolean_query(bq, inverted_index, all_doc_ids))
        print(f"\nBoolean Query: '{bq}' -> {len(matches)} matches")
        for doc_id in matches[:5]:
            print(f"  - [{doc_id}] {doc_titles[doc_id]}")

    print("\n" + "=" * 65)
    print("=== STEP 5: TF-IDF & COSINE SIMILARITY RANKING (3 QUERIES) ===")
    ranked_queries = [
        "diabetes symptoms treatment",
        "prevent heart disease high blood pressure",
        "genetic changes inherited condition",
    ]
    for rq in ranked_queries:
        results = ranker.search(rq, top_k=5)
        print(f"\nRanked Query: '{rq}'")
        for item in results:
            print(
                f"  Rank {item['rank']} | {item['doc_id']} | "
                f"Score: {item['score']:.4f} | {item['title']}"
            )


def main() -> None:
    dataset_path = "dataset/documents.csv"
    if not os.path.exists(dataset_path):
        print("Building cleaned MedQuAD dataset...")
        build_medquad_dataset(output_path=dataset_path, sample_size=500, random_state=42)

    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} documents from '{dataset_path}'.")

    # Build Inverted Index (Boolean branch)
    inverted_index, all_doc_ids, doc_titles = build_inverted_index(df)
    print(f"Built inverted index with {len(inverted_index)} unique stemmed terms.")

    # Fit TF-IDF Vectorizer (Ranked branch)
    ranker = TFIDFRankedRetriever(remove_stopwords=True)
    ranker.fit(df)
    print("Fitted TF-IDF vectorizer on document corpus.")

    while True:
        print("\n--- MedQuAD Information Retrieval System (Phase 1) ---")
        print("1. Run Full Phase 1 Demo (Steps 3, 4, and 5)")
        print("2. Boolean Search (AND / OR / NOT)")
        print("3. Ranked Search (TF-IDF + Cosine Similarity)")
        print("4. Exit")
        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            run_demo_suite(df, inverted_index, all_doc_ids, doc_titles, ranker)

        elif choice == "2":
            query = input("Enter Boolean query (e.g., diabetes AND treatment): ").strip()
            if not query:
                print("Empty query entered. Please type a valid Boolean query.")
                continue
            try:
                matches = sort_doc_ids(evaluate_boolean_query(query, inverted_index, all_doc_ids))
                print(f"Found {len(matches)} matching document(s):")
                for doc_id in matches[:10]:
                    print(f"  - [{doc_id}] {doc_titles[doc_id]}")
            except ValueError as err:
                print(f"Query Error: {err}")

        elif choice == "3":
            query = input("Enter search query (e.g., diabetes symptoms treatment): ").strip()
            if not query:
                print("Empty query entered. Please type a search query.")
                continue
            results = ranker.search(query, top_k=5)
            if not results:
                print("No matching documents found (score > 0).")
            else:
                print(f"Top {len(results)} Ranked Results:")
                for item in results:
                    print(
                        f"  Rank {item['rank']} | {item['doc_id']} | "
                        f"Score: {item['score']:.4f} | {item['title']}"
                    )

        elif choice == "4":
            print("Exiting MedQuAD IR System.")
            break
        else:
            print("Invalid selection. Choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
