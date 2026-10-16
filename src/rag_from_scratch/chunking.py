"""Day 03's structure-aware chunker, packaged for reuse by later days.

The code is the one taught in notebooks/day_03_chunking_and_structure.ipynb (sections 1, 3 and 4):
split a Document body at "## " headings, never cut inside a paragraph, list item or table, pack
whole units up to a word limit, split an oversize table between rows with its header row
repeated, and put a "Title > Section" context header on every chunk. Chunk IDs come from the
heading (doc_id#section-slug, with -part-N when a section is split).

One change in shape: the notebook read its limit from a global MAX_WORDS; here it is the default
argument max_words=80, the same value. Day 03's fixed-size windows are not copied, because no
later day needs them yet.

tests/test_shared_code.py runs the Day 03 notebook's own code and checks that this module returns
identical chunks.

Standard library only (plus sha256 from Day 02's ingest module).
"""

import re
from dataclasses import dataclass

from rag_from_scratch.ingest import sha256

MAX_WORDS = 80                        # Day 03's limit for one chunk, in words

WORD = re.compile(r"\w[^\s|]*")       # starts with a letter or digit; "|", "##" and ">" are not words
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def count_words(text: str) -> int:
    return len(WORD.findall(text))


def is_table_row(line: str) -> bool:
    return " | " in line              # Day 02's convention for table rows


def split_sections(body: str) -> list[tuple[str, list[str]]]:
    # Day 01's rule: "## " starts a section; text before the first heading becomes "Overview",
    # so nothing is silently dropped.
    sections, heading, lines = [], "Overview", []
    for line in body.splitlines():
        if line.startswith("## "):
            if lines:
                sections.append((heading, lines))
            heading, lines = line[3:].strip(), []
        elif line.strip():
            lines.append(line)
    if lines:
        sections.append((heading, lines))
    return sections


@dataclass(frozen=True)
class Chunk:
    chunk_id: str                  # stable, citable ID
    doc_id: str
    section: str | None            # heading the text came from
    text: str                      # exactly what is indexed, retrieved and shown to a reader
    # Inherited from the Document, so every retrieved chunk can be filtered, cited and dated
    title: str
    allowed_groups: tuple[str, ...]
    source_path: str
    last_reviewed: str | None
    doc_hash: str                  # the document version this chunk was cut from
    chunk_hash: str                # this chunk's own version


def make_chunk(doc, chunk_id: str, section: str | None, text: str) -> Chunk:
    return Chunk(chunk_id, doc.doc_id, section, text, doc.title, doc.allowed_groups, doc.source_path,
                 doc.last_reviewed, doc.content_hash, sha256(text))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def to_units(lines: list[str]) -> list[list[str]]:
    # Units that must not be cut: a run of table rows is one unit; any other line
    # (a paragraph or a list item) is a unit of its own.
    units: list[list[str]] = []
    for line in lines:
        if is_table_row(line) and units and is_table_row(units[-1][-1]):
            units[-1].append(line)
        else:
            units.append([line])
    return units


def split_unit(unit: list[str], room: int) -> list[list[str]]:
    # Split a unit that is too big on its own at its natural seams.
    if is_table_row(unit[0]):                     # a table: between rows, header repeated in every piece
        header, pieces = unit[0], [[unit[0]]]
        for row in unit[1:]:
            if len(pieces[-1]) > 1 and count_words("\n".join(pieces[-1] + [row])) > room:
                pieces.append([header])
            pieces[-1].append(row)
        return pieces
    sentences = [[]]                              # a paragraph: between sentences
    for sentence in SENTENCE_END.split(unit[0]):
        if sentences[-1] and count_words(" ".join(sentences[-1] + [sentence])) > room:
            sentences.append([])
        sentences[-1].append(sentence)
    return [[" ".join(s)] for s in sentences]     # a single sentence longer than `room` stays whole


def pack(units: list[list[str]], room: int) -> list[list[str]]:
    # Fill pieces with whole units, up to `room` words each.
    pieces, current = [], []
    for unit in units:
        if current and count_words("\n".join(current + unit)) > room:
            pieces.append(current)
            current = []
        if count_words("\n".join(unit)) <= room:
            current += unit
        else:
            *full, current = split_unit(unit, room)
            pieces += full
    if current:
        pieces.append(current)
    return pieces


def section_chunks(doc, max_words: int = MAX_WORDS, context_header: bool = True) -> list[Chunk]:
    chunks = []
    for heading, lines in split_sections(doc.body):
        header = f"{doc.title} > {heading}" if context_header else ""
        pieces = pack(to_units(lines), room=max_words - count_words(header))
        for n, piece in enumerate(pieces, start=1):
            suffix = f"-part-{n}" if len(pieces) > 1 else ""
            text = "\n".join(([header] if header else []) + piece)
            chunks.append(make_chunk(doc, f"{doc.doc_id}#{slugify(heading)}{suffix}", heading, text))
    return chunks
