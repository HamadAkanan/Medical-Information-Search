# Medical Information Search Engine (MedQuAD IR System — Phase 1)

**Course:** Information Retrieval Systems (902333) — Al al-Bayt University  
**Team Members:** Hamad Azzam Aknan, Elias Mohammad Alrefaii  
**Dataset:** MedQuAD (`prithvi1029/medquad-medical-qa`)

## Project Structure
- `prepare_dataset.py`: Cleans `medquad-cleaned-qa.csv`, removes duplicates/nulls, and builds `dataset/documents.csv` (500 reproducible Q&A records).
- `preprocessing.py`: Lowercasing, regex tokenization, configurable stopword filtering, and Porter stemming.
- `inverted_index.py`: Native Python dictionary-and-set inverted index (`term -> {doc_id}`).
- `boolean_search.py`: Exact set-based `AND` (intersection), `OR` (union), and `NOT` (complement) retrieval.
- `ranked_search.py`: Fitted `TfidfVectorizer` and cosine similarity top-k ranking.
- `main.py`: Integrated console application for running demonstrations and interactive queries.

## Exact Commands to Run
```bash
pip install -r requirements.txt
python prepare_dataset.py
python main.py
```
