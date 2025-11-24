def split_text(
    text: str, chunk_size: int = 1500, chunk_overlap: int = 300
) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # If this chunk reaches the end of the text, we are done.
        if end >= len(text):
            break
        start += chunk_size - chunk_overlap

    return chunks
