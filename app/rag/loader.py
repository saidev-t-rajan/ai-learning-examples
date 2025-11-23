import pypdf
import os
import re


class DocumentLoader:
    def load(self, file_path: str) -> str:
        """
        Loads a file (PDF or TXT) and extracts all text content.
        Cleans the text by replacing single newlines with spaces (unwrapping).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        text = ""
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        elif file_path.endswith(".pdf"):
            text_content = []
            try:
                reader = pypdf.PdfReader(file_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                text = "\n\n".join(text_content)
            except Exception as e:
                raise Exception(f"Failed to read PDF: {e}")

        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        """
        Replaces single newlines with spaces, but keeps paragraph breaks (double newlines).
        """
        # 1. Replace 3+ newlines with 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 2. Replace single newlines that are NOT preceded or followed by another newline with a space
        # Negative lookbehind (?<!\n) and negative lookahead (?!\n)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        return text
