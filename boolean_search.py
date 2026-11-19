from preprocessing import preprocess_text


def _term_postings(
    term: str,
    inverted_index: dict[str, set[str]],
) -> set[str]:
    """Preprocess a single query term and return its matching doc_id set."""
    processed = preprocess_text(term, remove_stopwords=False)
    if not processed:
        return set()
    return set(inverted_index.get(processed[0], set()))


def boolean_and(
    terms: list[str],
    inverted_index: dict[str, set[str]],
) -> set[str]:
    """Return intersection of postings lists for all given terms."""
    cleaned_terms = [t for t in terms if preprocess_text(t, remove_stopwords=False)]
    if not cleaned_terms:
        return set()

    result = _term_postings(cleaned_terms[0], inverted_index)
    for term in cleaned_terms[1:]:
        result = result & _term_postings(term, inverted_index)
    return result


def boolean_or(
    terms: list[str],
    inverted_index: dict[str, set[str]],
) -> set[str]:
    """Return union of postings lists for all given terms."""
    result: set[str] = set()
    for term in terms:
        result = result | _term_postings(term, inverted_index)
    return result


def boolean_not(
    exclude_term: str,
    inverted_index: dict[str, set[str]],
    all_doc_ids: set[str],
    include_term: str | None = None,
) -> set[str]:
    """
    Return set complement against all_doc_ids (NOT exclude_term),
    or (include_term AND NOT exclude_term) when include_term is provided.
    """
    excluded_docs = _term_postings(exclude_term, inverted_index)
    if include_term:
        included_docs = _term_postings(include_term, inverted_index)
        return included_docs - excluded_docs
    return all_doc_ids - excluded_docs


def evaluate_boolean_query(
    query: str,
    inverted_index: dict[str, set[str]],
    all_doc_ids: set[str],
) -> set[str]:
    """
    Parse and evaluate a Boolean query supporting AND, OR, NOT, and AND NOT.
    Example queries:
      - 'diabetes AND treatment'
      - 'diabetes OR insulin'
      - 'diabetes AND NOT gestational'
      - 'NOT diabetes'
    """
    if not query or not query.strip():
        return set()

    tokens = query.strip().split()
    upper_tokens = [t.upper() for t in tokens]

    # Case 1: Leading unary NOT (e.g., "NOT diabetes")
    if upper_tokens[0] == "NOT" and len(tokens) == 2:
        return boolean_not(tokens[1], inverted_index, all_doc_ids)

    # Case 2: Binary expression with AND NOT (e.g., "diabetes AND NOT gestational")
    if len(tokens) == 4 and upper_tokens[1] == "AND" and upper_tokens[2] == "NOT":
        return boolean_not(tokens[3], inverted_index, all_doc_ids, include_term=tokens[0])

    # Case 3: Chained AND query (e.g., "diabetes AND symptoms AND treatment")
    if "AND" in upper_tokens and "OR" not in upper_tokens and "NOT" not in upper_tokens:
        terms = [t for t in tokens if t.upper() != "AND"]
        return boolean_and(terms, inverted_index)

    # Case 4: Chained OR query (e.g., "diabetes OR cancer")
    if "OR" in upper_tokens and "AND" not in upper_tokens and "NOT" not in upper_tokens:
        terms = [t for t in tokens if t.upper() != "OR"]
        return boolean_or(terms, inverted_index)

    # Case 5: Single keyword lookup
    if len(tokens) == 1:
        return _term_postings(tokens[0], inverted_index)

    raise ValueError(
        "Unsupported Boolean syntax. Use 'term1 AND term2', 'term1 OR term2', "
        "'term1 AND NOT term2', or 'NOT term1'."
    )
