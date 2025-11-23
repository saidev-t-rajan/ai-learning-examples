from typing import List


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks of maximum size `chunk_size` with `chunk_overlap`.
        """
        separator = self.separators[-1]

        # Find the appropriate separator to use
        for sep in self.separators:
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                break

        # Split the text using the separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Now go through splits and merge them or recurse
        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if separator:
                    # Recurse with remaining separators
                    next_seps = self.separators[self.separators.index(separator) + 1 :]
                    if next_seps:
                        sub_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=self.chunk_size,
                            chunk_overlap=self.chunk_overlap,
                            separators=next_seps,
                        )
                        good_splits.extend(sub_splitter.split_text(s))
                    else:
                        # No separators left, force split (slicing)
                        good_splits.extend(self._force_split(s))
                else:
                    good_splits.extend(self._force_split(s))

        return self._merge_splits(good_splits, separator)

    def _force_split(self, text: str) -> List[str]:
        """Splits text by character if no separators work."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merges small splits into chunks of max size."""
        docs = []
        current_doc = []
        current_length = 0
        separator_len = len(separator)

        for s in splits:
            s_len = len(s)

            # If adding this split exceeds chunk_size (considering separator)
            if (
                current_length + s_len + (separator_len if current_doc else 0)
                > self.chunk_size
            ):
                if current_doc:
                    docs.append(separator.join(current_doc))

                    # Handle overlap: remove from start until it fits or is empty
                    # Ideally we want to keep the last X chars roughly = chunk_overlap
                    # But we are working with "atomic splits", so we pop atoms.

                    while current_length > self.chunk_overlap:
                        popped = current_doc.pop(0)
                        current_length -= len(popped) + separator_len
                        # Corner case: if popping made it empty but current_length still "wrong" (e.g. 0)
                        if not current_doc:
                            current_length = 0
                            break

            current_doc.append(s)
            current_length += s_len + (separator_len if len(current_doc) > 1 else 0)

        if current_doc:
            docs.append(separator.join(current_doc))

        return docs
