"""Deterministic rule-based toxicity auditor and confidential data leak prevention filter."""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# Standard categories of sensitive, abusive, or restricted expressions
TOXICITY_PATTERNS = {
    "profanity": [
        r"\b(?:damn|hell|crap|idiot|stupid)\b",
    ],
    "hostility": [
        r"\b(?:hate\s+you|kill\s+yourself|destroy\s+you|threaten)\b",
    ],
    "confidential_disclosure": [
        r"(?:bearer\s+[a-zA-Z0-9_\-\.]{20,})",
        r"(?:api[_-]?key\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"])",
        r"(?:password\s*[:=]\s*['\"][^'\"]{6,}['\"])",
        r"(?:AKIA[0-9A-Z]{16})",
        r"(?:ghp_[a-zA-Z0-9]{36})",
    ],
}


def _scan_text(text: str) -> Dict[str, Any]:
    """Inspect text for toxic tokens, hate speech triggers, and leaked secrets."""
    matches: List[Dict[str, Any]] = []

    for category, patterns in TOXICITY_PATTERNS.items():
        for pat in patterns:
            compiled = re.compile(pat, flags=re.IGNORECASE)
            for m in compiled.finditer(text):
                matched_val = m.group(0)
                # Redact matched text if it is a credential
                display_val = matched_val if category != "confidential_disclosure" else "[REDACTED_SECRET]"
                matches.append({
                    "category": category,
                    "matched": display_val,
                    "start": m.start(),
                    "end": m.end(),
                })

    is_clean = len(matches) == 0
    return {
        "is_clean": is_clean,
        "violation_count": len(matches),
        "violations": matches,
    }


def execute_toxic_language_filter(
    text: Optional[str] = None,
    input_path: Optional[str] = None,
    redact: bool = False,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Scan text or documents for offensive language and sensitive credential leaks.

    Args:
        text: Raw text string to analyze.
        input_path: Optional path to target document.
        redact: Produce sanitized copy with detected violations masked.
        output_path: Optional destination to write JSON audit report.

    Returns:
        Structured dictionary reporting safety audit metrics.
    """
    if input_path and os.path.isfile(input_path):
        with open(input_path, mode="r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    elif text is not None:
        content = text
    else:
        raise ValueError("Either text or input_path must be specified.")

    scan = _scan_text(content)
    sanitized_text = content

    if redact and not scan["is_clean"]:
        # Sort in reverse order of position to replace cleanly
        for v in sorted(scan["violations"], key=lambda x: x["start"], reverse=True):
            s = v["start"]
            e = v["end"]
            sanitized_text = sanitized_text[:s] + f"[{v['category'].upper()}_REDACTED]" + sanitized_text[e:]

    result = {
        "status": "success",
        "is_clean": scan["is_clean"],
        "violation_count": scan["violation_count"],
        "violations": scan["violations"],
        "sanitized_text": sanitized_text if redact else None,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, mode="w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=2)

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Deterministic rule-based toxicity auditor and confidential leak prevention filter."
    )
    parser.add_argument("--text", "-t", help="Raw input string")
    parser.add_argument("--input", "-i", help="Path to input document")
    parser.add_argument("--redact", "-r", action="store_true", help="Redact identified violations")
    parser.add_argument("--output", "-o", help="Path to write JSON audit report")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        res = execute_toxic_language_filter(
            text=args.text,
            input_path=args.input,
            redact=args.redact,
            output_path=args.output,
        )
        status_str = "CLEAN" if res["is_clean"] else f"VIOLATIONS DETECTED ({res['violation_count']})"
        print(f"Safety Scan Result: {status_str}")
        for v in res["violations"]:
            print(f"  - [{v['category']}] {v['matched']}")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
