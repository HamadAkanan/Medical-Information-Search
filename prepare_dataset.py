import os
import re
import pandas as pd


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace/newlines into single spaces."""
    return re.sub(r"\s+", " ", str(text)).strip()


def build_medquad_dataset(
    raw_csv_path: str = "medquad-cleaned-qa.csv",
    output_path: str = "dataset/documents.csv",
    sample_size: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load MedQuAD CSV, remove missing/empty and duplicate records,
    sample a reproducible subset (300-1,000 recommended), and standardize fields.
    """
    df = pd.read_csv(raw_csv_path)

    # Verify required columns exist
    required_cols = {"Question", "Answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Expected columns {required_cols}, got {list(df.columns)}")

    # 1. Drop missing values and strip whitespace
    df = df.dropna(subset=["Question", "Answer"]).copy()
    df["Question"] = df["Question"].apply(normalize_whitespace)
    df["Answer"] = df["Answer"].apply(normalize_whitespace)
    df["category"] = (
        df["qtype"].fillna("General").apply(normalize_whitespace)
        if "qtype" in df.columns
        else "Medical QA"
    )

    # 2. Remove empty records and exact Question+Answer duplicates
    df = df[(df["Question"] != "") & (df["Answer"] != "")]
    df = df.drop_duplicates(subset=["Question", "Answer"]).reset_index(drop=True)

    # 3. Reproducible random sample (minimum 100 required; 300-1,000 recommended)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    # 4. Standardize schema: doc_id, title, text (Question + Answer), category
    clean_df = pd.DataFrame({
        "doc_id": [f"D{i + 1}" for i in range(len(df))],
        "title": df["Question"],
        "text": df["Question"] + " " + df["Answer"],
        "category": df["category"],
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_df.to_csv(output_path, index=False, encoding="utf-8")
    return clean_df


if __name__ == "__main__":
    result = build_medquad_dataset()
    print(f"Successfully created 'dataset/documents.csv' with {len(result)} unique documents.")
