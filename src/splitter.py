"""Custom Markdown-aware text splitter.

Strategy:
1. Split the document into sections by Markdown headings (#, ##, ###, ...).
   Every section remembers its heading path, e.g.
   "Дебетові картки МіБанку > Мі Gold".
2. A section that fits into ``max_chars`` becomes one chunk.
3. A longer section is split into blocks (paragraphs, lists, tables are kept whole),
   which are packed greedily into chunks of at most ``max_chars``. The tail of
   the previous chunk (up to ``overlap_chars``) is repeated at the start of the
   next chunk so that context is not lost at the boundary.
4. A single block that is still too long is split by sentences, then by hard cut.

The heading path is prepended to each chunk's text, so the embedding "knows"
which product/section the chunk belongs to (e.g. a bare list of fees becomes
"Мі Gold: ...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class Chunk:
    id: str
    text: str  # text that is embedded and shown to the LLM (with heading path)
    source: str  # file name
    heading: str  # heading path, e.g. "Депозити > Мі Скарбничка"
    index: int  # position of the chunk within its file
    metadata: dict = field(default_factory=dict)


@dataclass
class _Section:
    heading_path: list[str]
    lines: list[str]


def _split_sections(text: str) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    current = _Section(heading_path=[], lines=[])
    in_code = False

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else HEADING_RE.match(line)
        if match:
            sections.append(current)
            level, title = len(match.group(1)), match.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current = _Section(heading_path=[t for _, t in stack], lines=[])
        else:
            current.lines.append(line)
    sections.append(current)

    return [s for s in sections if "\n".join(s.lines).strip()]


def _split_blocks(body: str) -> list[str]:
    """Blank-line separated blocks: paragraphs, whole lists, whole tables."""
    return [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]


def _split_long_block(block: str, max_chars: int) -> list[str]:
    """Split an oversized block by lines/sentences, hard-cut as a last resort."""
    units: list[str] = []
    for line in block.splitlines():
        if len(line) <= max_chars:
            units.append(line)
            continue
        for sentence in SENTENCE_RE.split(line):
            while len(sentence) > max_chars:
                units.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if sentence:
                units.append(sentence)

    pieces, buf = [], ""
    for unit in units:
        candidate = f"{buf}\n{unit}" if buf else unit
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            pieces.append(buf)
            buf = unit
    if buf:
        pieces.append(buf)
    return pieces


def _tail(blocks: list[str], overlap_chars: int) -> list[str]:
    """Last blocks of a chunk whose total length fits into ``overlap_chars``."""
    tail: list[str] = []
    size = 0
    for block in reversed(blocks):
        if size + len(block) > overlap_chars:
            break
        tail.insert(0, block)
        size += len(block)
    return tail


def _pack(blocks: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    units: list[str] = []
    for block in blocks:
        units.extend([block] if len(block) <= max_chars else _split_long_block(block, max_chars))

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and len("\n\n".join(current + [unit])) > max_chars:
            chunks.append("\n\n".join(current))
            current = _tail(current, overlap_chars)
            # Drop overlap if it would not leave room for the new unit.
            if len("\n\n".join(current + [unit])) > max_chars:
                current = []
        current.append(unit)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def split_markdown(
    text: str,
    source: str,
    max_chars: int = 1000,
    overlap_chars: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in _split_sections(text):
        heading = " > ".join(section.heading_path)
        body = "\n".join(section.lines).strip()
        prefix = f"{heading}\n\n" if heading else ""
        budget = max(200, max_chars - len(prefix))
        for piece in _pack(_split_blocks(body), budget, overlap_chars):
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{source}::{index}",
                    text=prefix + piece,
                    source=source,
                    heading=heading,
                    index=index,
                )
            )
    return chunks
