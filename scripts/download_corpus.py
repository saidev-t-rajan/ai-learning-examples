import os
import requests
import time

# Directory to save files
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "corpus_large"
)

# Public domain books (Project Gutenberg)
# Using text files as our loader supports them and they are easier to download reliably than PDFs
BOOKS = {
    # Original set
    "frankenstein.txt": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "pride_and_prejudice.txt": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "sherlock_holmes.txt": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "moby_dick.txt": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "dracula.txt": "https://www.gutenberg.org/cache/epub/345/pg345.txt",
    "ulysses.txt": "https://www.gutenberg.org/cache/epub/4300/pg4300.txt",
    "war_and_peace.txt": "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
    "count_of_monte_cristo.txt": "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
    # Additions for size (>50MB total target)
    "les_miserables.txt": "https://www.gutenberg.org/cache/epub/135/pg135.txt",
    "don_quixote.txt": "https://www.gutenberg.org/cache/epub/996/pg996.txt",
    "brothers_karamazov.txt": "https://www.gutenberg.org/cache/epub/28054/pg28054.txt",
    "anna_karenina.txt": "https://www.gutenberg.org/cache/epub/1399/pg1399.txt",
    "david_copperfield.txt": "https://www.gutenberg.org/cache/epub/766/pg766.txt",
    "bleak_house.txt": "https://www.gutenberg.org/cache/epub/1023/pg1023.txt",
    "middlemarch.txt": "https://www.gutenberg.org/cache/epub/145/pg145.txt",
    "three_musketeers.txt": "https://www.gutenberg.org/cache/epub/1257/pg1257.txt",
    "crime_and_punishment.txt": "https://www.gutenberg.org/cache/epub/2554/pg2554.txt",
    "great_expectations.txt": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
    "wealth_of_nations.txt": "https://www.gutenberg.org/cache/epub/3300/pg3300.txt",
    "critique_of_pure_reason.txt": "https://www.gutenberg.org/cache/epub/4280/pg4280.txt",
    "leviathan.txt": "https://www.gutenberg.org/cache/epub/3207/pg3207.txt",
    "republic.txt": "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
    "odyssey.txt": "https://www.gutenberg.org/cache/epub/1727/pg1727.txt",
    "iliad.txt": "https://www.gutenberg.org/cache/epub/6130/pg6130.txt",
    "tom_jones.txt": "https://www.gutenberg.org/cache/epub/6593/pg6593.txt",
    "vanity_fair.txt": "https://www.gutenberg.org/cache/epub/599/pg599.txt",
    "woman_in_white.txt": "https://www.gutenberg.org/cache/epub/583/pg583.txt",
    "clarissa.txt": "https://www.gutenberg.org/cache/epub/5299/pg5299.txt",
    "tale_of_two_cities.txt": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    "grimms_fairy_tales.txt": "https://www.gutenberg.org/cache/epub/2591/pg2591.txt",
    "leaves_of_grass.txt": "https://www.gutenberg.org/cache/epub/1322/pg1322.txt",
    "secret_garden.txt": "https://www.gutenberg.org/cache/epub/113/pg113.txt",
    "peter_pan.txt": "https://www.gutenberg.org/cache/epub/16/pg16.txt",
    "alice_in_wonderland.txt": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "through_the_looking_glass.txt": "https://www.gutenberg.org/cache/epub/12/pg12.txt",
    "treasure_island.txt": "https://www.gutenberg.org/cache/epub/120/pg120.txt",
    "jekyll_and_hyde.txt": "https://www.gutenberg.org/cache/epub/43/pg43.txt",
    "jungle_book.txt": "https://www.gutenberg.org/cache/epub/236/pg236.txt",
    "metamorphosis.txt": "https://www.gutenberg.org/cache/epub/5200/pg5200.txt",
    "dorian_gray.txt": "https://www.gutenberg.org/cache/epub/174/pg174.txt",
    "huckleberry_finn.txt": "https://www.gutenberg.org/cache/epub/76/pg76.txt",
    "tom_sawyer.txt": "https://www.gutenberg.org/cache/epub/74/pg74.txt",
    "prince_and_pauper.txt": "https://www.gutenberg.org/cache/epub/1837/pg1837.txt",
    "connecticut_yankee.txt": "https://www.gutenberg.org/cache/epub/86/pg86.txt",
    "life_on_the_mississippi.txt": "https://www.gutenberg.org/cache/epub/245/pg245.txt",
    "innocents_abroad.txt": "https://www.gutenberg.org/cache/epub/3176/pg3176.txt",
    "roughing_it.txt": "https://www.gutenberg.org/cache/epub/3177/pg3177.txt",
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
