"""LLM token usage and context window budget estimator across model architectures."""

import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional

# Standard context windows and character-to-token ratios
MODEL_PROFILES = {
    "llama3-8b": {"context_window": 8192, "char_per_token": 3.8, "overhead_per_message": 3},
    "llama3.1-8b": {"context_window": 131072, "char_per_token": 3.8, "overhead_per_message": 3},
    "mistral-7b": {"context_window": 32768, "char_per_token": 3.9, "overhead_per_message": 3},
    "gpt-4o": {"context_window": 128000, "char_per_token": 4.0, "overhead_per_message": 4},
    "gpt-4": {"context_window": 8192, "char_per_token": 4.0, "overhead_per_message": 4},
    "claude-3.5-sonnet": {"context_window": 200000, "char_per_token": 3.7, "overhead_per_message": 3},
}


def _estimate_tokens(text: str, char_per_token: float = 3.8) -> int:
    """Estimate token count based on whitespace, punctuation, and character heuristics."""
    # Split on words and punctuation
    tokens = re.findall(r"\w+|[^\w\s]", text)
    by_tokens = len(tokens)
    by_chars = len(text) / char_per_token
    # Weighted average of lexical items and character length
    return max(1, int(math.ceil((by_tokens * 0.4) + (by_chars * 0.6))))


def execute_token_budget_estimator(
    text: Optional[str] = None,
    input_path: Optional[str] = None,
    target_model: str = "llama3-8b",
    reserved_output_tokens: int = 2048,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate token count and remaining headroom within LLM context boundaries.

    Args:
        text: Raw text string to analyze.
        input_path: Optional path to text file.
        target_model: Model profile key.
        reserved_output_tokens: Allocation buffer for model response.
        output_path: Optional destination to write JSON budget report.

    Returns:
        Structured dictionary reporting token counts and headroom percentage.
    """
    if input_path and os.path.isfile(input_path):
        with open(input_path, mode="r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    elif text is not None:
        content = text
    else:
        raise ValueError("Either text or input_path must be specified.")

    model_key = target_model.lower()
    profile = MODEL_PROFILES.get(model_key, MODEL_PROFILES["llama3-8b"])

    estimated_input_tokens = _estimate_tokens(content, profile["char_per_token"])
    total_budget = profile["context_window"]
    available_headroom = total_budget - estimated_input_tokens - reserved_output_tokens
    fits_in_context = available_headroom >= 0
    utilization_pct = round((estimated_input_tokens / total_budget) * 100, 2)

    result = {
        "status": "success",
        "model": target_model,
        "context_window_limit": total_budget,
        "character_count": len(content),
        "estimated_prompt_tokens": estimated_input_tokens,
        "reserved_completion_tokens": reserved_output_tokens,
        "remaining_headroom_tokens": max(0, available_headroom),
        "utilization_percent": utilization_pct,
        "fits_in_context": fits_in_context,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, mode="w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=2)

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM token usage and context window budget estimator."
    )
    parser.add_argument("--text", "-t", help="Raw text string to estimate")
    parser.add_argument("--input", "-i", help="Path to text document")
    parser.add_argument(
        "--model",
        "-m",
        choices=list(MODEL_PROFILES.keys()),
        default="llama3-8b",
        help="Target LLM model architecture",
    )
    parser.add_argument(
        "--reserve",
        "-r",
        type=int,
        default=2048,
        help="Tokens reserved for output generation",
    )
    parser.add_argument("--output", "-o", help="Path to write JSON output")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        res = execute_token_budget_estimator(
            text=args.text,
            input_path=args.input,
            target_model=args.model,
            reserved_output_tokens=args.reserve,
            output_path=args.output,
        )
        fit_str = "FITS" if res["fits_in_context"] else "EXCEEDS BUDGET"
        print(f"Model {res['model']}: {res['estimated_prompt_tokens']} prompt tokens ({res['utilization_percent']}% utilized, {fit_str})")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
