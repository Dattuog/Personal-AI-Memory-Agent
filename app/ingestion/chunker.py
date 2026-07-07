import re


def _token_count(text: str) -> int:
    return len(text.split())


def _split_long_sentence(sentence: str, max_tokens: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


def chunk_text(text: str, max_tokens: int = 300, overlap: int = 50) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]
    expanded: list[str] = []
    for sentence in sentences:
        if _token_count(sentence) > max_tokens:
            expanded.extend(_split_long_sentence(sentence, max_tokens))
        else:
            expanded.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in expanded:
        sentence_tokens = _token_count(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current))
            overlap_words = " ".join(current).split()[-overlap:] if overlap > 0 else []
            current = [" ".join(overlap_words)] if overlap_words else []
            current_tokens = len(overlap_words)
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks
