#!/usr/bin/env python3
"""
VYREX AI — OANC OUP candidate integrity audit
READ-ONLY: never edits source text or metadata.
Run from the project root with:
    python scripts/oanc_integrity_audit.py
"""
from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OANC_ROOT = PROJECT / "data" / "raw" / "oanc" / "extracted" / "OANC-GrAF" / "data" / "written_2" / "non-fiction" / "OUP"
PROFILE_CSV = PROJECT / "reports" / "oanc_oup_language_profiles.csv"
REPORT_DIR = PROJECT / "reports"

# These are screening patterns, not automatic proof of an error.
SUSPICIOUS_PATTERNS = {
    "aect_family": re.compile(r"\b\w*aect\w*\b", re.IGNORECASE),
    "eect_family": re.compile(r"\b\w*eect\w*\b", re.IGNORECASE),
    "dicult_family": re.compile(r"\b\w*dicult\w*\b", re.IGNORECASE),
    "dier_family": re.compile(r"\b\w*dier\w*\b", re.IGNORECASE),
    "eort_family": re.compile(r"\b\w*eort\w*\b", re.IGNORECASE),
    "scaold_family": re.compile(r"\b\w*scaold\w*\b", re.IGNORECASE),
    "ecturing_family": re.compile(r"\b\w*ectur\w*\b", re.IGNORECASE),
    "missing_initial_capital_candidate": re.compile(r"(?<![A-Za-z])ectur(?:e|ing|ed)\b", re.IGNORECASE),
}
PLACEHOLDER_RE = re.compile(r"\b(?:TODO|FIXME|TBD|PLACEHOLDER|INSERT HERE)\b|FIELD\s*\(", re.IGNORECASE)
WORD_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"‘(])")

def excerpt(text: str, pos: int, radius: int = 75) -> str:
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    return re.sub(r"\s+", " ", text[start:end]).strip()

def main() -> int:
    if not PROFILE_CSV.exists():
        print(f"ERROR: profile CSV not found: {PROFILE_CSV}")
        print("Place oanc_oup_language_profiles.csv in the project's reports folder.")
        return 2
    if not OANC_ROOT.exists():
        print(f"ERROR: OANC OUP folder not found: {OANC_ROOT}")
        print("Check that OANC-GrAF was extracted to the expected location.")
        return 2

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with PROFILE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        profile_rows = list(csv.DictReader(f))

    summary_rows = []
    detail_rows = []
    print("=" * 72)
    print("VYREX AI — OANC OUP CANDIDATE INTEGRITY AUDIT")
    print("=" * 72)
    print("MODE: READ-ONLY")
    print(f"PROFILE RECORDS: {len(profile_rows)}")
    print(f"SOURCE ROOT: {OANC_ROOT}")
    print()

    for record in profile_rows:
        relative = Path(record["file"])
        source = OANC_ROOT / relative
        if not source.exists():
            summary_rows.append({
                "file": record["file"], "status": "MISSING_SOURCE",
                "characters": 0, "words": 0, "replacement_chars": 0,
                "control_chars": 0, "placeholder_matches": 0,
                "duplicate_nonempty_lines": 0, "duplicate_paragraphs": 0,
                "very_long_lines": 0, "suspicious_pattern_hits": 0,
                "metadata_file_exists": False, "text_ends_with_sentence_punctuation": False,
                "first_text": "", "last_text": "", "notes": "Source text file not found"
            })
            continue

        raw = source.read_bytes()
        # Decode strictly first; report decoding problems without silently replacing bytes.
        decode_error = ""
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            decode_error = f"UTF8_DECODE_ERROR:{exc.start}-{exc.end}"
            text = raw.decode("utf-8", errors="replace")

        words = WORD_RE.findall(text)
        lines = text.splitlines()
        nonempty_lines = [line.strip() for line in lines if line.strip()]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        line_counts = Counter(nonempty_lines)
        paragraph_counts = Counter(re.sub(r"\s+", " ", p).strip() for p in paragraphs)

        replacement_count = text.count("\ufffd")
        control_count = sum(1 for ch in text if unicodedata.category(ch) == "Cc" and ch not in "\n\r\t")
        placeholder_matches = list(PLACEHOLDER_RE.finditer(text))
        duplicate_lines = sum(n - 1 for n in line_counts.values() if n > 1)
        duplicate_paragraphs = sum(n - 1 for n in paragraph_counts.values() if n > 1)
        very_long_lines = sum(1 for line in lines if len(line) > 1000)

        pattern_hits = 0
        for label, pattern in SUSPICIOUS_PATTERNS.items():
            matches = list(pattern.finditer(text))
            pattern_hits += len(matches)
            for match in matches:
                detail_rows.append({
                    "file": record["file"],
                    "finding_type": label,
                    "matched_text": match.group(0),
                    "context": excerpt(text, match.start()),
                    "review_note": "Screening hit only; inspect context before deciding"
                })

        for match in placeholder_matches:
            detail_rows.append({
                "file": record["file"], "finding_type": "placeholder",
                "matched_text": match.group(0), "context": excerpt(text, match.start()),
                "review_note": "Check whether this is an unresolved editorial placeholder"
            })

        for match in re.finditer("\ufffd", text):
            detail_rows.append({
                "file": record["file"], "finding_type": "replacement_character",
                "matched_text": "U+FFFD", "context": excerpt(text, match.start()),
                "review_note": "Do not guess the missing character; compare authoritative source"
            })

        # A single paragraph may simply reflect source formatting; report it, do not fail it.
        first_text = re.sub(r"\s+", " ", text[:220]).strip()
        last_text = re.sub(r"\s+", " ", text[-220:]).strip()
        trimmed = text.rstrip()
        ends_sentence = bool(trimmed) and trimmed[-1] in ".!?”’\"')]}"

        anc_file = source.with_suffix(".anc")
        notes = []
        if decode_error:
            notes.append(decode_error)
        if replacement_count:
            notes.append("contains U+FFFD")
        if pattern_hits:
            notes.append("spelling-screen hits require manual review")
        if len(paragraphs) <= 1:
            notes.append("one detected paragraph; verify source line-break structure")
        if not ends_sentence:
            notes.append("file does not end with common sentence punctuation")
        if not anc_file.exists():
            notes.append("matching .anc metadata missing")

        summary_rows.append({
            "file": record["file"],
            "status": "REVIEW" if notes else "NO_AUTOMATED_FLAGS",
            "characters": len(text),
            "words": len(words),
            "replacement_chars": replacement_count,
            "control_chars": control_count,
            "placeholder_matches": len(placeholder_matches),
            "duplicate_nonempty_lines": duplicate_lines,
            "duplicate_paragraphs": duplicate_paragraphs,
            "very_long_lines": very_long_lines,
            "suspicious_pattern_hits": pattern_hits,
            "metadata_file_exists": anc_file.exists(),
            "text_ends_with_sentence_punctuation": ends_sentence,
            "first_text": first_text,
            "last_text": last_text,
            "notes": "; ".join(notes)
        })

    summary_path = REPORT_DIR / "oanc_oup_integrity_summary.csv"
    details_path = REPORT_DIR / "oanc_oup_integrity_findings.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)
    with details_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["file", "finding_type", "matched_text", "context", "review_note"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    flagged = sum(1 for row in summary_rows if row["status"] == "REVIEW")
    print(f"DOCUMENTS CHECKED: {len(summary_rows)}")
    print(f"DOCUMENTS FLAGGED FOR REVIEW: {flagged}")
    print(f"DETAIL FINDINGS: {len(detail_rows)}")
    print()
    print("REPORTS CREATED:")
    print(f"  {summary_path}")
    print(f"  {details_path}")
    print()
    print("IMPORTANT:")
    print("- No source .txt or .anc files were changed.")
    print("- Pattern hits are clues, not confirmed spelling errors.")
    print("- A clean automated result is not final corpus approval.")
    print("- Training was NOT started.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
