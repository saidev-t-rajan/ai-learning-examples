import pytest
from app.rag.loader import load_document
from app.rag.splitter import split_text

# --- Test DocumentLoader ---


def test_document_loader_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_document("non_existent_file.pdf")


def test_document_loader_extracts_pdf_text(tmp_path):
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

    text = load_document(str(pdf_path))

    # pypdf should extract "Hello World"
    assert "Hello World" in text


def test_document_loader_extracts_txt_text(tmp_path):
    # Create a real TXT file
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("This is a simple text file.", encoding="utf-8")

    text = load_document(str(txt_path))

    assert "This is a simple text file." in text


# --- Test TextSplitter ---


def test_splitter_simple_no_split():
    """Test that text smaller than chunk_size is returned as a single chunk."""
    text = "12345"
    chunks = split_text(text, chunk_size=10, chunk_overlap=0)
    assert chunks == ["12345"]


def test_splitter_exact_split():
    """Test splitting exactly at the chunk size."""
    text = "1234567890"
    chunks = split_text(text, chunk_size=5, chunk_overlap=0)
    assert chunks == ["12345", "67890"]


def test_splitter_with_overlap():
    """Test splitting with overlap."""
    text = "123456789"
    # Chunk 1: 0-5 -> "12345"
    # Step: 5 - 2 = 3
    # Chunk 2: 3-8 -> "45678"
    # Step: 3 + 3 = 6
    # Chunk 3: 6-11 -> "789" (remainder)
    chunks = split_text(text, chunk_size=5, chunk_overlap=2)
    assert chunks == ["12345", "45678", "789"]


def test_splitter_does_not_infinite_loop():
    """Ensure no infinite loop if overlap is too large."""
    # If overlap >= chunk_size, we might get stuck if we aren't careful.
    # Our implementation uses `start += chunk_size - chunk_overlap`
    text = "12345678901"  # 11 chars
    # C1: 0-10 -> "1234567890"
    # Step: 10 - 9 = 1
    # C2: 1-11 -> "2345678901"
    chunks = split_text(text, chunk_size=10, chunk_overlap=9)
    assert len(chunks) == 2
    assert chunks[0] == "1234567890"
    assert chunks[1] == "2345678901"
