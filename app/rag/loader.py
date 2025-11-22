import pypdf
import os


class PDFLoader:
    def load(self, file_path: str) -> str:
        """
        Loads a PDF file and extracts all text content.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: The extracted text content.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: If there is an error reading the PDF.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        text_content = []
        try:
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
            return "\n".join(text_content)
        except Exception as e:
            raise Exception(f"Failed to read PDF: {e}")
