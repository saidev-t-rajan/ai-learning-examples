import pytest
from app.rag.loader import PDFLoader
from app.rag.splitter import RecursiveCharacterTextSplitter

# --- Test PDFLoader ---


def test_pdf_loader_file_not_found():
    loader = PDFLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("non_existent_file.pdf")


def test_pdf_loader_extracts_text(tmp_path):
    # Create a real PDF file
    pdf_path = tmp_path / "test.pdf"

    # Minimal PDF with "Hello World"
    # This is binary data.
    minimal_pdf_content = (
        b"%PDF-1.1\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"5 0 obj\n<< /Length 44 >>\nstream\nBT /F1 24 Tf 100 700 Td (Hello World) Tj ET\nendstream\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n0000000230 00000 n \n0000000318 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n412\n%%EOF\n"
    )

    pdf_path.write_bytes(minimal_pdf_content)

    loader = PDFLoader()
    text = loader.load(str(pdf_path))

    # pypdf should extract "Hello World"
    assert "Hello World" in text


# --- Test RecursiveCharacterTextSplitter ---


def test_splitter_simple_split():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10, chunk_overlap=0, separators=[" "]
    )
    text = "abc def ghi"
    chunks = splitter.split_text(text)
    # Expect: "abc", "def", "ghi" -> but merge?
    # "abc" (3) -> fit
    # "abc def" (7) -> fit
    # "abc def ghi" (11) -> too big
    # So it should contain "abc def", then overlap?
    # Logic: "abc def" (7). Next is "ghi". 7+3+1 = 11 > 10.
    # Emit "abc def".
    # Current doc becomes empty (overlap 0).
    # Add "ghi".
    assert chunks == ["abc def", "ghi"]


def test_splitter_with_overlap():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10, chunk_overlap=4, separators=[" "]
    )
    text = "abc def ghi jkl"
    # "abc def" (7) -> fits
    # + "ghi" (3) -> 7+1+3 = 11 > 10.
    # Emit "abc def".
    # Overlap: "def" (3) -> keep? 3 <= 4.
    # Current = ["def"].
    # Add "ghi" -> "def ghi" (7).
    # + "jkl" (3) -> 7+1+3 = 11 > 10.
    # Emit "def ghi".
    # Overlap: "ghi" (3).
    # Current = ["ghi"].
    # Add "jkl" -> "ghi jkl" (7).
    # Emit "ghi jkl".
    chunks = splitter.split_text(text)
    assert chunks == ["abc def", "def ghi", "ghi jkl"]


def test_splitter_recursive():
    # Test where it must go down to characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5, chunk_overlap=1, separators=[" "]
    )
    text = "abcdefgh"  # No spaces, 8 chars
    chunks = splitter.split_text(text)
    # Should force split: "abcde", "fgh" (with overlap?)
    # "abcde" (5)
    # Overlap 1: "e"
    # "efgh" (4)
    # Wait, my force_split logic:
    # range(0, 8, 5-1=4) -> 0, 4
    # 0:5 -> "abcde"
    # 4:9 -> "efgh"
    assert chunks == ["abcde", "efgh"]


def test_splitter_separators():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=20, chunk_overlap=0, separators=["\n\n", "\n"]
    )
    text = "Para1.\n\nPara2 is longer.\nIt has two lines."
    # Para1. (6)
    # Para2 is longer.\nIt has two lines. (16+1+17 = 34) -> Too big.
    # Split by \n: "Para2 is longer." (16), "It has two lines." (17).
    # All chunks < 20?
    # "Para1." (6) -> chunk
    # "Para2 is longer." (16) -> chunk
    # "It has two lines." (17) -> chunk
    chunks = splitter.split_text(text)
    assert len(chunks) == 3
    assert chunks[0] == "Para1."
    assert chunks[1] == "Para2 is longer."
    assert chunks[2] == "It has two lines."
