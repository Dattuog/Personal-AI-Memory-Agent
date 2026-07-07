from app.ingestion.chunker import chunk_text


def test_chunk_sizes_stay_under_max_tokens() -> None:
    text = " ".join(f"word{i}." for i in range(120))
    chunks = chunk_text(text, max_tokens=30, overlap=5)
    assert chunks
    assert all(len(chunk.split()) <= 30 for chunk in chunks)


def test_overlap_and_no_data_loss_for_multi_chunk_text() -> None:
    sentences = [f"sentence {i} has several words." for i in range(20)]
    chunks = chunk_text(" ".join(sentences), max_tokens=20, overlap=4)
    assert len(chunks) > 1
    assert chunks[0].split()[-4:] == chunks[1].split()[:4]
    compact_original = " ".join(" ".join(sentences).split())
    for sentence in sentences:
        assert sentence in compact_original
        assert sentence in " ".join(chunks)
