"""Polymorphic prompt generator with recursive nested spintax expansion and Jaccard uniqueness scoring."""

import argparse
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Set


def _expand_spintax_once(text: str) -> str:
    """Resolve the innermost spintax group {option1|option2|...}."""
    pattern = re.compile(r"\{([^{}]+)\}")
    while True:
        match = pattern.search(text)
        if not match:
            break
        options = match.group(1).split("|")
        choice = random.choice(options)
        text = text[: match.start()] + choice + text[match.end() :]
    return text


def _calculate_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity index based on word token sets."""
    tokens_a = set(re.findall(r"\w+", text_a.lower()))
    tokens_b = set(re.findall(r"\w+", text_b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return round(intersection / union, 4)


def execute_smart_prompt_spintax_generator(
    template: str,
    count: int = 5,
    min_uniqueness: float = 0.2,
    seed: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate diverse prompt variations using nested spintax patterns.

    Args:
        template: Input template containing nested spintax expressions (e.g. '{Write|Draft} a {clean|robust} script').
        count: Desired number of distinct variations.
        min_uniqueness: Minimum average Jaccard distance required.
        seed: Optional RNG seed for deterministic reproduction.
        output_path: Optional path to write output JSON.

    Returns:
        Structured dictionary listing generated variations and uniqueness score.
    """
    if seed is not None:
        random.seed(seed)

    variations: List[str] = []
    seen = set()
    max_attempts = count * 20
    attempts = 0

    while len(variations) < count and attempts < max_attempts:
        attempts += 1
        expanded = _expand_spintax_once(template).strip()
        if expanded not in seen:
            seen.add(expanded)
            variations.append(expanded)

    # Compute pairwise Jaccard diversity
    diversity_scores: List[float] = []
    for i in range(len(variations)):
        for j in range(i + 1, len(variations)):
            sim = _calculate_jaccard_similarity(variations[i], variations[j])
            diversity_scores.append(1.0 - sim)

    avg_uniqueness = round(sum(diversity_scores) / len(diversity_scores), 4) if diversity_scores else 1.0

    result = {
        "status": "success",
        "template": template,
        "requested_count": count,
        "generated_count": len(variations),
        "average_uniqueness": avg_uniqueness,
        "uniqueness_threshold_met": avg_uniqueness >= min_uniqueness,
        "variations": variations,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, mode="w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=2)

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Polymorphic prompt generator with nested spintax and Jaccard diversity scoring."
    )
    parser.add_argument("--template", "-t", required=True, help="Input spintax template text or file path")
    parser.add_argument("--count", "-c", type=int, default=5, help="Number of variations to produce")
    parser.add_argument("--seed", "-s", type=int, help="Optional RNG seed")
    parser.add_argument("--output", "-o", help="Path to write JSON output")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        tmpl = args.template
        if os.path.isfile(tmpl):
            with open(tmpl, "r", encoding="utf-8") as f:
                tmpl = f.read()

        res = execute_smart_prompt_spintax_generator(
            template=tmpl,
            count=args.count,
            seed=args.seed,
            output_path=args.output,
        )
        print(f"Generated {res['generated_count']} variations (Average Uniqueness: {res['average_uniqueness'] * 100:.1f}%):")
        for idx, var in enumerate(res["variations"], 1):
            print(f"  {idx}. {var}")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
