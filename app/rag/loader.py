import pypdf
import re
from app.core.utils import validate_file_path


def load_document(file_path: str) -> str:
    """
    Loads a file (PDF or TXT) and extracts all text content.
    Cleans the text by replacing single newlines with spaces (unwrapping).
    """
    valid_path = validate_file_path(file_path, allowed_extensions=[".txt", ".pdf"])

    text = ""
    # validate_file_path ensures the suffix is correct, but we check again for dispatch
    if valid_path.suffix.lower() == ".txt":
        text = _load_txt(str(valid_path))

    elif valid_path.suffix.lower() == ".pdf":
        text = _load_pdf(str(valid_path))

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
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = [para.replace("\n", " ") for para in paragraphs]
    return "\n\n".join(cleaned_paragraphs)
