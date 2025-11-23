import os
import requests
import time

# Directory to save files
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "corpus")

# Public domain books (Project Gutenberg)
# Using text files as our loader supports them and they are easier to download reliably than PDFs
BOOKS = {
    "frankenstein.txt": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "pride_and_prejudice.txt": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "sherlock_holmes.txt": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "moby_dick.txt": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "dracula.txt": "https://www.gutenberg.org/cache/epub/345/pg345.txt",
    "ulysses.txt": "https://www.gutenberg.org/cache/epub/4300/pg4300.txt",
    "war_and_peace.txt": "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
    # Requested specific PDF files
    "lotr_fellowship.pdf": "https://www.mrsmuellersworld.com/uploads/1/3/0/5/13054185/lord-of-the-rings-01-the-fellowship-of-the-ring_full_text.pdf",
    # NOTE: The requested Scribd link ("https://www.scribd.com/document/346643809/Revenge-Of-The-Sith-pdf")
    # returns an HTML page, not a raw PDF, preventing automated ingestion.
    # We substitute it with another large public domain text to maintain corpus size.
    # "star_wars_revenge_sith.pdf": "https://www.scribd.com/document/346643809/Revenge-Of-The-Sith-pdf",
    "count_of_monte_cristo.txt": "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
}


def download_file(url, filename):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filename}, already exists.")
        return filepath

    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        print(f"Saved to {filepath}")
        return filepath
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return None


def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f"Downloading corpus to {DATA_DIR}...")

    files_downloaded = 0
    total_size = 0

    for filename, url in BOOKS.items():
        filepath = download_file(url, filename)
        if filepath:
            files_downloaded += 1
            total_size += os.path.getsize(filepath)
            # Be nice to Gutenberg servers
            time.sleep(1)

    print("-" * 30)
    print("Download complete.")
    print(f"Files: {files_downloaded}")
    print(f"Total Size: {total_size / (1024 * 1024):.2f} MB")
    print("\nYou can now ingest these files using the CLI:")
    print("python -m app.main")
    print("/ingest data/corpus/frankenstein.txt")


if __name__ == "__main__":
    main()
