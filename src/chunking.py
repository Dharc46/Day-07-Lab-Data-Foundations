from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Split text into sentences using lookbehinds
        sentences = re.split(r'(?<=\. )|(?<=\! )|(?<=\? )|(?<=\.\n)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i:i + self.max_sentences_per_chunk]
            chunk_text = " ".join(group).strip()
            if chunk_text:
                chunks.append(chunk_text)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        if not remaining_separators:
            # Fallback when no separators left
            return [current_text[i:i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]
        
        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]
        
        if separator != "" and separator not in current_text:
            return self._split(current_text, next_separators)
        
        if separator == "":
            parts = list(current_text)
        else:
            parts = current_text.split(separator)
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        for part in parts:
            part_len = len(part)
            if part_len > self.chunk_size:
                if current_chunk:
                    joined = separator.join(current_chunk)
                    chunks.extend(self._split(joined, next_separators))
                    current_chunk = []
                    current_length = 0
                chunks.extend(self._split(part, next_separators))
            else:
                sep_len = len(separator) if current_chunk else 0
                if current_length + part_len + sep_len <= self.chunk_size:
                    current_chunk.append(part)
                    current_length += part_len + sep_len
                else:
                    if current_chunk:
                        joined = separator.join(current_chunk)
                        chunks.append(joined)
                    current_chunk = [part]
                    current_length = part_len
                    
        if current_chunk:
            joined = separator.join(current_chunk)
            chunks.append(joined)
            
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_prod = sum(x * y for x, y in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(y * y for y in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_prod / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)
        
        strategies = {
            'fixed_size': fixed_chunker.chunk(text),
            'by_sentences': sentence_chunker.chunk(text),
            'recursive': recursive_chunker.chunk(text)
        }
        
        comparison = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            comparison[name] = {
                'count': count,
                'avg_length': avg_length,
                'chunks': chunks
            }
        return comparison
