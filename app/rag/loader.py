import pypdf
import os
import re


def load_document(file_path: str) -> str:
    """
    Loads a file (PDF or TXT) and extracts all text content.
    Cleans the text by replacing single newlines with spaces (unwrapping).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    text = ""
    if file_path.endswith(".txt"):
        text = _load_txt(file_path)

    elif file_path.endswith(".pdf"):
        text = _load_pdf(file_path)

    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    return _clean_text(text)


def _load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _load_pdf(file_path: str) -> str:
    text_content = []
    reader = pypdf.PdfReader(file_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_content.append(page_text)
    return "\n\n".join(text_content)


def _clean_text(text: str) -> str:
    """
    Replaces single newlines with spaces, but keeps paragraph breaks (double newlines).
    """
    # 1. Replace 3+ newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 2. Split by paragraph (double newline)
    paragraphs = text.split("\n\n")

    # 3. Unwrap lines within each paragraph
    cleaned_paragraphs = [para.replace("\n", " ") for para in paragraphs]

    # 4. Rejoin paragraphs
    return "\n\n".join(cleaned_paragraphs)
