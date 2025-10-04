"""Data extraction pipeline for Bengali and English legal PDFs.

This module provides utilities to extract raw text from PDF documents, parse
relevant metadata (case numbers, dates, and party names), and persist the
results as UTF-8 encoded JSON files.  It is designed to be resilient when
processing large collections of documents while preserving Bengali Unicode
characters.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Configure a module-level logger. Consumers can override the configuration from
# their own entrypoints if desired.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PartyMetadata:
    """Structured representation of party name findings."""

    versus_pairs: List[Tuple[str, str]]
    role_based: Dict[str, List[str]]


@dataclass
class DocumentExtraction:
    """Container for a single PDF's extracted artefacts."""

    source_pdf: Path
    text: str
    case_numbers: List[str]
    case_types: List[str]
    dates: List[str]
    judgments: List[str]
    party_metadata: PartyMetadata

    def to_json_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation of the extraction."""

        return {
            "source_pdf": str(self.source_pdf),
            "text": self.text,
            "metadata": {
                "case_numbers": self.case_numbers,
                "case_types": self.case_types,
                "dates": self.dates,
                "judgments": self.judgments,
                "party_names": {
                    "versus_pairs": [
                        {"first_party": first.strip(), "second_party": second.strip()}
                        for first, second in self.party_metadata.versus_pairs
                    ],
                    "roles": self.party_metadata.role_based,
                },
            },
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

MONTH_LOOKUP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def ordered_unique(items: Iterable[str]) -> List[str]:
    """Return list with duplicates removed while preserving original order."""

    seen = set()
    unique_items: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract Unicode text from a PDF file using pdfminer."""

    from pdfminer.high_level import extract_text  # Local import to avoid hard dependency at module import time

    text = extract_text(str(pdf_path))
    # Normalise non-breaking spaces and stray control characters that often
    # appear in scraped PDFs.
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\u200b\ufeff]", "", text)
    return text


def extract_case_numbers(text: str) -> List[str]:
    """Identify likely case numbers using a collection of regular expressions."""

    patterns = [
        re.compile(
            r"\b(?:Case|Appeal|Petition|Writ|Civil|Criminal|Review|Reference|Suit|Revision)\s*(?:No\.?|Number)\s*[:\-]?\s*([A-Za-z0-9./-]+)",
            re.IGNORECASE,
        ),
        re.compile(r"\bNo\.?\s*[:\-]?\s*([0-9]{1,5}\s*of\s*[12][0-9]{3})", re.IGNORECASE),
        re.compile(r"\b([0-9]{1,5}\/[12][0-9]{3})\b"),
        re.compile(r"\b([A-Z]{1,4}\s*[0-9]{1,5}\s*of\s*[12][0-9]{3})\b", re.IGNORECASE),
    ]

    matches: List[str] = []
    for pattern in patterns:
        matches.extend(match.strip() for match in pattern.findall(text))
    return ordered_unique(matches)


def extract_case_types(text: str) -> List[str]:
    """Extract the case type descriptors associated with the matter."""

    # Common English descriptors
    english_patterns = [
        r"\b(Criminal\s+Appeal)",
        r"\b(Civil\s+Appeal)",
        r"\b(Criminal\s+Revision)",
        r"\b(Civil\s+Revision)",
        r"\b(Civil\s+Petition)",
        r"\b(Review\s+Petition)",
        r"\b(Review\s+Case)",
        r"\b(Writ\s+Petition)",
        r"\b(Reference\s+Case)",
        r"\b(Misc(?:ellaneous)?\.?\s+Case)",
        r"\b(Special\s+Leave\s+Petition)",
        r"\b(Execution\s+Case)",
        r"\b(Company\s+Petition)",
        r"\b(Arbitration\s+Case)",
        r"\b(Income\s+Tax\s+Reference)",
        r"\b(Family\s+Suit)",
    ]

    # Bengali descriptors frequently seen in legal documents
    bengali_patterns = [
        r"(ফৌজদারি\s+আপিল)",
        r"(দেওয়ানি\s+মামলা)",
        r"(রিট\s+পিটিশন)",
        r"(রিভিউ\s+মামলা)",
        r"(মিস\.?\s+কেস)",
        r"(ক্রিমিনাল\s+রিভিশন)",
    ]

    generic_pattern = re.compile(
        r"\b((?:[A-Z][A-Za-z.&/\-]*\s+){0,3}(?:Appeal|Case|Petition|Application|Writ|Reference|Revision|Suit))\s*(?=No\.?|Number|\d)",
        re.IGNORECASE,
    )

    matches: List[str] = []
    for pattern in english_patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    for pattern in bengali_patterns:
        matches.extend(re.findall(pattern, text))

    matches.extend(match.strip() for match in generic_pattern.findall(text))

    # Normalise spacing and casing for English entries.
    normalised: List[str] = []
    for match in matches:
        cleaned = re.sub(r"\s+", " ", match).strip()
        if cleaned:
            if re.search(r"[A-Za-z]", cleaned):
                cleaned = cleaned.title()
            normalised.append(cleaned)

    return ordered_unique(normalised)


def _normalise_year(year: str) -> int:
    year_int = int(year)
    if year_int < 100:
        # Assume 2000s for two-digit years >= 50? We'll use heuristic: >= 50 -> 1900s else 2000s.
        return 1900 + year_int if year_int >= 50 else 2000 + year_int
    return year_int


def _format_date(year: int, month: int, day: int) -> Optional[str]:
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def extract_dates(text: str) -> List[str]:
    """Extract dates in ISO format from textual and numeric representations."""

    numeric_pattern = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b")
    textual_pattern = re.compile(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sept\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)\s*,?\s*(\d{4})\b",
        re.IGNORECASE,
    )

    matches: List[str] = []
    for day, month, year in numeric_pattern.findall(text):
        iso = _format_date(_normalise_year(year), int(month), int(day))
        if iso:
            matches.append(iso)

    for day, month_text, year in textual_pattern.findall(text):
        month_key = month_text.lower().rstrip('.')
        month_num = MONTH_LOOKUP.get(month_key)
        if month_num is None:
            continue
        iso = _format_date(int(year), month_num, int(day))
        if iso:
            matches.append(iso)

    return ordered_unique(matches)


def extract_party_names(text: str) -> PartyMetadata:
    """Extract party names from the body of text using heuristics."""

    # Versus patterns capture constructions like "A vs B" or "A v. B"
    versus_pattern = re.compile(
        r"^\s*(?P<first>[^\n]+?)\s+(?:v\.?|vs\.?|versus)\s+(?P<second>[^\n]+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    versus_pairs = [
        (match.group("first").strip(), match.group("second").strip())
        for match in versus_pattern.finditer(text)
    ]

    role_pattern = re.compile(
        r"^\s*(Petitioner|Respondent|Appellant|Defendant|Plaintiff|Complainant|Opposite Party)s?\s*[:.-]?\s*(.+)$",
        re.IGNORECASE | re.MULTILINE,
    )

    role_map = {
        "petitioner": "petitioners",
        "respondent": "respondents",
        "appellant": "appellants",
        "defendant": "defendants",
        "plaintiff": "plaintiffs",
        "complainant": "complainants",
        "opposite party": "opposite_parties",
    }

    role_based: Dict[str, List[str]] = {value: [] for value in role_map.values()}

    for role, name in role_pattern.findall(text):
        key = role_map.get(role.lower())
        if not key:
            continue
        cleaned_name = name.strip()
        if cleaned_name:
            bucket = role_based.setdefault(key, [])
            if cleaned_name not in bucket:
                bucket.append(cleaned_name)

    # Remove keys with no values to keep JSON compact.
    role_based = {key: value for key, value in role_based.items() if value}

    return PartyMetadata(versus_pairs=versus_pairs, role_based=role_based)


def extract_judgments(text: str) -> List[str]:
    """Extract judgement outcomes or dispositive language from the text."""

    patterns = [
        re.compile(
            r"\b(?:the\s+)?(appeal|petition|case|suit|application)\s+(?:is|was|be|stands)\s+(partly\s+allowed|partially\s+allowed|allowed|dismissed|disposed\s+of|rejected|withdrawn|maintained|quashed|granted|denied|remanded|set\s+aside)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(appeal|petition|case|suit|application)\s+(allowed|dismissed|disposed\s+of|withdrawn|rejected|granted|denied)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(order|judgment|decree)\s+(?:is|was|stands|hereby)?\s*(upheld|affirmed|confirmed|set\s+aside)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(bail|anticipatory\s+bail)\s+(?:is|was|be|stands)?\s*(granted|rejected|allowed|denied)\b",
            re.IGNORECASE,
        ),
    ]

    judgments: List[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            groups = [group for group in match.groups() if group]
            if not groups:
                continue
            description = " ".join(groups)
            description = re.sub(r"\s+", " ", description).strip()
            if description:
                judgments.append(description.capitalize())

    return ordered_unique(judgments)


def assemble_document_extraction(pdf_path: Path, text: str) -> DocumentExtraction:
    """Build a :class:`DocumentExtraction` from raw text."""

    metadata = PartyMetadata(versus_pairs=[], role_based={})
    try:
        metadata = extract_party_names(text)
    except Exception as exc:
        logger.warning("Failed to parse party names for %s: %s", pdf_path.name, exc)

    case_numbers = extract_case_numbers(text)
    case_types = extract_case_types(text)
    dates = extract_dates(text)
    judgments = extract_judgments(text)

    return DocumentExtraction(
        source_pdf=pdf_path,
        text=text,
        case_numbers=case_numbers,
        case_types=case_types,
        dates=dates,
        judgments=judgments,
        party_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# File-system facing helpers
# ---------------------------------------------------------------------------


def iter_pdf_files(path: Path) -> Iterator[Path]:
    """Yield PDF files from a path (single file or directory)."""

    if path.is_file() and path.suffix.lower() == ".pdf":
        yield path
        return

    if path.is_dir():
        for pdf_file in sorted(path.rglob("*.pdf")):
            if pdf_file.is_file():
                yield pdf_file
        return

    raise FileNotFoundError(f"No PDF files found at {path}")


def write_json(output_path: Path, data: Dict[str, object]) -> None:
    """Persist JSON using UTF-8 encoding without ASCII escaping."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def process_pdfs(input_path: Path, output_dir: Path, overwrite: bool = False) -> List[Path]:
    """Process PDFs and write JSON outputs.

    Returns the list of generated JSON file paths.
    """

    created_files: List[Path] = []
    for pdf_file in iter_pdf_files(input_path):
        try:
            logger.info("Processing %s", pdf_file)
            text = extract_text_from_pdf(pdf_file)
            extraction = assemble_document_extraction(pdf_file, text)

            output_path = output_dir / (pdf_file.stem + ".json")
            if output_path.exists() and not overwrite:
                logger.info("Skipping existing output %s", output_path)
                continue

            write_json(output_path, extraction.to_json_dict())
            created_files.append(output_path)
            logger.info("Wrote %s", output_path)
        except Exception as exc:
            logger.error("Failed to process %s: %s", pdf_file, exc)
    return created_files


def configure_logging(verbosity: int) -> None:
    """Set up logging level based on CLI flags."""

    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract text and metadata from Bengali and English legal PDFs.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a PDF file or a directory containing PDF files.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Directory where JSON outputs will be stored.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON outputs if they already exist.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (can be used multiple times).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    created_files = process_pdfs(args.input, args.output, overwrite=args.overwrite)
    if not created_files:
        logger.warning("No JSON files were generated. Check input path or enable --overwrite if files exist.")


if __name__ == "__main__":
    main()
