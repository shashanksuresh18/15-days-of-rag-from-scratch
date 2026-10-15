"""Day 02's ingestion pipeline, packaged for reuse by later days.

This is the code taught step by step in notebooks/day_02_ingestion_and_metadata.ipynb, copied
without behavioural changes: 13 sample files in, 11 `Document` records plus 2 `Quarantined`
entries out. Read the notebook to learn how it works; import this module when a later day
just needs the records.

tests/test_shared_code.py runs the Day 02 notebook's own code and checks that this module
returns identical records, so the copy cannot drift silently.

Standard library plus pypdf.
"""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import pypdf
from pypdf import PdfReader

SNAPSHOT_DATE = date(2026, 10, 13)  # when the sample export was taken; review ages are measured from here
MAX_REVIEW_AGE_DAYS = 365           # an older review date gets a "review overdue" warning
PDF_MODE = "layout"                 # pypdf extraction mode; Day 02 section 3.3 shows why not "plain"


# ----------------------------------------------------------------------------- locating files
def find_repo_root(start: Path | None = None) -> Path:
    # Walk up until data/sample/acme-handbook is found, so code works from the repo root
    # or from notebooks/. No hard-coded paths.
    start = (start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        if (folder / "data" / "sample" / "acme-handbook").is_dir():
            return folder
    raise FileNotFoundError(
        "Could not find data/sample/acme-handbook. Start Jupyter from inside the cloned repository."
    )


def list_files(folder: Path) -> list[Path]:
    # Sorted for the same order on every OS; hidden files such as .DS_Store are skipped.
    return sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))


# ----------------------------------------------------------------------------- the contract
@dataclass(frozen=True)
class Document:
    # Identity
    doc_id: str                       # stable: the same file gets the same ID on every run
    # Descriptive metadata, read from the document (useful, but any author can edit it)
    title: str
    owner: str | None
    last_reviewed: str | None         # ISO date "YYYY-MM-DD", or None if missing or unreadable
    # Access metadata, from the system of record, never from the document's own text
    allowed_groups: tuple[str, ...]
    # Provenance, added by the pipeline
    source_path: str                  # relative to the repository root
    source_format: str                # "md", "html", "pdf" or "txt"
    parser: str                       # which code and library version produced `body`
    content_hash: str                 # SHA-256 of `body`: identifies this *version* of the text
    # Content: normalised text with "## " section headings, whatever the source format
    body: str
    warnings: tuple[str, ...] = ()    # problems worth fixing that don't block indexing


@dataclass(frozen=True)
class Quarantined:
    # A file the pipeline refused to index, and why. Reported, never silently dropped.
    source_path: str
    reasons: tuple[str, ...]


@dataclass
class Parsed:
    # What each format adapter returns: raw descriptive metadata plus body text.
    title: str | None
    owner: str | None
    last_reviewed: str | None
    body: str
    parser: str
    notes: tuple[str, ...] = ()       # adapter observations, e.g. "1 image, no text layer"


# ----------------------------------------------------------------------------- normalisation
def normalise(text: str) -> str:
    # Make text from every format comparable: one Unicode form, one line ending, tidy spaces.
    text = unicodedata.normalize("NFKC", text)        # e.g. ligature "ﬁ" -> "fi", no-break space -> space
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def space_headings(text: str) -> str:
    # One blank line before every heading, as in the Markdown sources.
    return re.sub(r"\n+(#{2,3} )", r"\n\n\1", text)


# ----------------------------------------------------------------------------- Markdown
def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    # Split a '---'-delimited 'key: value' header from the Markdown body.
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    metadata: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return metadata, "\n".join(lines[i + 1:]).strip()
        key, sep, value = line.partition(":")
        if sep:
            metadata[key.strip()] = value.strip()
    raise ValueError("Front matter opened with '---' but never closed.")


def parse_markdown(path: Path) -> Parsed:
    metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
    return Parsed(metadata.get("title"), metadata.get("owner"), metadata.get("last_reviewed"),
                  body, parser="front matter + Markdown (standard library)")


# ----------------------------------------------------------------------------- HTML
SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript", "button", "form"}
BLOCK_TAGS = {"h1", "h2", "h3", "p", "li", "ol", "ul", "table", "tr", "td", "th", "br", "div", "section"}


class PageTextParser(HTMLParser):
    # Keeps readable text inside <main> as Markdown-style lines and reads <meta name=... content=...>.
    # Text outside <main>, or inside SKIP_TAGS, goes to `dropped` so you can audit what was removed.

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)       # "&copy;" -> "©"
        self.meta: dict[str, str] = {}
        self.title_tag, self.h1 = "", ""
        self.lines: list[str] = []
        self.dropped: list[str] = []
        self._text: list[str] = []                    # text of the block being read
        self._cells: list[str] | None = None          # cells of the table row being read
        self._lists: list[list] = []                  # [kind, counter] for each open <ol>/<ul>
        self._in_title = False
        self._main_depth = self._skip_depth = 0

    def _keeping(self) -> bool:
        return self._main_depth > 0 and self._skip_depth == 0

    def _take_text(self) -> str:
        text, self._text = " ".join("".join(self._text).split()), []
        return text

    def _flush(self, prefix: str = "") -> None:
        if text := self._take_text():
            self.lines.append(prefix + text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and attrs.get("name"):
            self.meta[attrs["name"].lower()] = (attrs.get("content") or "").strip()
        elif tag == "title":
            self._in_title = True
        elif tag == "main":
            self._main_depth += 1
        elif tag in SKIP_TAGS:
            self._skip_depth += 1
        elif tag in ("ol", "ul"):
            self._lists.append([tag, 0])
        elif tag == "li" and self._lists:
            self._lists[-1][1] += 1
        elif tag == "tr":
            self._cells = []
        if tag in BLOCK_TAGS and self._keeping():
            self._flush()                             # loose text before a block becomes its own line

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "main":
            self._main_depth -= 1
        elif tag in SKIP_TAGS:
            self._skip_depth -= 1
        if not self._keeping():
            return
        if tag == "h1":
            self.h1 = self._take_text()
        elif tag in ("h2", "h3"):
            self._flush("## " if tag == "h2" else "### ")
        elif tag == "li":
            kind, n = self._lists[-1] if self._lists else ("ul", 0)
            self._flush(f"{n}. " if kind == "ol" else "- ")
        elif tag in ("td", "th") and self._cells is not None:
            self._cells.append(self._take_text())
        elif tag == "tr" and self._cells is not None:
            self.lines.append(" | ".join(self._cells))
            self._cells = None
        elif tag in ("ol", "ul") and self._lists:
            self._lists.pop()
        elif tag == "p":
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title_tag += data
        elif self._keeping():
            self._text.append(data)
        elif data.strip():
            self.dropped.append(" ".join(data.split()))


def read_html(path: Path) -> PageTextParser:
    page = PageTextParser()
    page.feed(path.read_text(encoding="utf-8"))
    page.close()
    return page


def parse_html(path: Path) -> Parsed:
    page = read_html(path)
    return Parsed(title=page.h1 or page.title_tag.strip() or None,   # <title> often carries a site suffix
                  owner=page.meta.get("owner"), last_reviewed=page.meta.get("last-reviewed"),
                  body="\n".join(page.lines), parser="html.parser, <main> only (standard library)")


# ----------------------------------------------------------------------------- PDF
TABLE_GAP = re.compile(r"\s{3,}")                         # layout mode pads table columns with spaces
NUMBERED_HEADING = re.compile(r"^\d{1,2}\.\s+([A-Z][^.]{0,60})$")   # "3. Daily limits by city"


def pdf_page_lines(reader: PdfReader, mode: str) -> list[list[str]]:
    # Non-empty text lines, one list per page.
    pages = []
    for page in reader.pages:
        if mode == "layout":
            text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False)
        else:
            text = page.extract_text()
        pages.append([line.rstrip() for line in text.splitlines() if line.strip()])
    return pages


def remove_running_lines(pages: list[list[str]]) -> tuple[list[list[str]], set[str]]:
    # Drop lines that appear on every page once digits are masked: headers, footers, page numbers.
    if len(pages) < 2:
        return pages, set()

    def mask(line: str) -> str:
        return re.sub(r"\d+", "#", " ".join(line.split()))

    on_every_page = set.intersection(*({mask(line) for line in page} for page in pages))
    return [[line for line in page if mask(line) not in on_every_page] for page in pages], on_every_page


def pdf_lines_to_body(lines: list[str], title: str | None) -> str:
    out: list[str] = []
    for line in (raw.strip() for raw in lines):
        previous = out[-1] if out else ""
        cells = TABLE_GAP.split(line)
        if not out and line == title:
            continue                                          # the big title line is metadata, not body
        if len(cells) > 1:
            out.append(" | ".join(cells))                     # 2. a table row
        elif (heading := NUMBERED_HEADING.match(line)):
            out.append(f"## {heading.group(1)}")              # 3. "1. Purpose" -> "## Purpose"
        elif previous.endswith("-") and previous[-2:-1].isalpha() and line[:1].islower():
            out[-1] = previous[:-1] + line                    # 4. "reim-" + "bursement"
        elif previous and not previous.startswith("#") and " | " not in previous:
            out[-1] = previous + " " + line                   # 4. same paragraph, next visual line
        else:
            out.append(line)
    return "\n".join(out)


def parse_pdf(path: Path, mode: str | None = None) -> Parsed:
    mode = mode or PDF_MODE
    reader = PdfReader(path)
    info = reader.metadata or {}
    title, parser = info.get("/Title"), f"pypdf {pypdf.__version__} ({mode} mode)"
    pages = pdf_page_lines(reader, mode)
    if not any(pages):                                        # no text layer at all
        images = sum(len(page.images) for page in reader.pages)
        note = f"{len(reader.pages)} page, {images} image, 0 characters of text"
        return Parsed(title, info.get("/Owner"), info.get("/LastReviewed"), "", parser, (note,))
    pages, running = remove_running_lines(pages)              # 1. headers and footers
    body = pdf_lines_to_body([line for page in pages for line in page], title)
    return Parsed(title, info.get("/Owner"), info.get("/LastReviewed"), body, parser,
                  (f"removed running lines: {sorted(running)}",) if running else ())


# ----------------------------------------------------------------------------- plain text
def parse_text(path: Path) -> Parsed:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    title = lines[0].strip() if lines else None               # a guess: plain text has no title field
    return Parsed(title, None, None, "\n".join(lines[1:]).strip(), parser="plain text (standard library)")


PARSERS = {".md": parse_markdown, ".html": parse_html, ".pdf": parse_pdf, ".txt": parse_text}


# ----------------------------------------------------------------------------- identity, validation
def make_doc_id(path: Path) -> str:
    return path.stem


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def load_permissions(acl_file: Path) -> dict[str, tuple[str, ...]]:
    export = json.loads(acl_file.read_text(encoding="utf-8"))
    return {path: tuple(sorted(groups)) for path, groups in export["permissions"].items()}


def check_review_date(value: str | None) -> tuple[str | None, list[str]]:
    if not value:
        return None, ["missing last_reviewed"]
    try:
        reviewed = date.fromisoformat(value) if ISO_DATE.fullmatch(value) else None
    except ValueError:
        reviewed = None
    if reviewed is None:
        return None, [f"unreadable last_reviewed {value!r} (expected YYYY-MM-DD; not guessed)"]
    age = (SNAPSHOT_DATE - reviewed).days
    if age > MAX_REVIEW_AGE_DAYS:
        return value, [f"review overdue: last reviewed {age} days before the snapshot"]
    return value, []


def validate(parsed: Parsed, groups: tuple[str, ...] | None) -> tuple[list[str], list[str], str | None]:
    problems, warnings = [], []
    if not parsed.body:
        detail = f" ({'; '.join(parsed.notes)})" if parsed.notes else ""
        problems.append(f"no extractable text{detail}: scanned? run OCR first")
    if not parsed.title:
        problems.append("no title")
    if groups is None:
        problems.append("no entry in the permission export: fail closed (visible to nobody)")
    elif not groups:
        problems.append("permission entry lists no groups: fail closed")
    if not parsed.owner:
        warnings.append("missing owner")
    last_reviewed, date_warnings = check_review_date(parsed.last_reviewed)
    return problems, warnings + date_warnings, last_reviewed


# ----------------------------------------------------------------------------- the pipeline
def ingest(root: Path) -> tuple[list[Document], list[Quarantined]]:
    corpus_dir = root / "data" / "sample" / "acme-handbook"
    permissions = load_permissions(root / "data" / "sample" / "access-control.json")
    documents, quarantined, seen = [], [], {}
    for path in list_files(corpus_dir):
        source_path = path.relative_to(root).as_posix()
        adapter = PARSERS.get(path.suffix.lower())
        if adapter is None:
            quarantined.append(Quarantined(source_path, (f"unsupported format {path.suffix!r}",)))
            continue
        try:
            parsed = adapter(path)
        except Exception as error:                            # one bad file must not stop the run
            quarantined.append(Quarantined(source_path, (f"could not parse: {type(error).__name__}: {error}",)))
            continue
        parsed.body = space_headings(normalise(parsed.body))
        groups = permissions.get(path.relative_to(corpus_dir.parent).as_posix())
        problems, warnings, last_reviewed = validate(parsed, groups)
        doc_id = make_doc_id(path)
        if doc_id in seen:
            problems.append(f"duplicate doc_id {doc_id!r}, already used by {seen[doc_id]}")
        if problems:
            quarantined.append(Quarantined(source_path, tuple(problems)))
            continue
        seen[doc_id] = source_path
        documents.append(Document(
            doc_id=doc_id, title=normalise(parsed.title), owner=parsed.owner, last_reviewed=last_reviewed,
            allowed_groups=groups, source_path=source_path, source_format=path.suffix[1:].lower(),
            parser=parsed.parser, content_hash=sha256(parsed.body), body=parsed.body,
            warnings=tuple(warnings),
        ))
    return documents, quarantined
