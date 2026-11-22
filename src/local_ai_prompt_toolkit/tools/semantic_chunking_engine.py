"""Semantic document chunking engine preserving heading boundaries and coherent paragraphs for RAG pipelines."""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


def _split_into_paragraphs(text: str) -> List[str]:
    """Split raw markdown or text into paragraphs while preserving structure."""
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


def execute_semantic_chunking_engine(
    input_path: Optional[str] = None,
    text: Optional[str] = None,
    max_chunk_size: int = 800,
    chunk_overlap: int = 100,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Segment large text or markdown files into coherent chunks respecting boundaries.

    Args:
        input_path: Path to target document.
        text: Raw document text string.
        max_chunk_size: Target maximum character limit per segment.
        chunk_overlap: Overlap characters between adjacent segments.
        output_path: Optional destination to write JSON chunks.

    Returns:
        Structured dictionary listing segmented chunks and boundary statistics.
    """
    if input_path and os.path.isfile(input_path):
        with open(input_path, mode="r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    elif text is not None:
        content = text
    else:
        raise ValueError("Either text or input_path must be specified.")

    paragraphs = _split_into_paragraphs(content)
    chunks: List[Dict[str, Any]] = []

    current_chunk = ""
    current_headings: List[str] = []
    chunk_index = 1

    for para in paragraphs:
        # Detect markdown heading
        if para.startswith("#"):
            heading_line = para.split("\n", 1)[0].strip()
            current_headings.append(heading_line)

        # Check if adding this paragraph exceeds maximum chunk size
        if current_chunk and (len(current_chunk) + len(para) + 2 > max_chunk_size):
            chunks.append({
                "chunk_id": chunk_index,
                "character_count": len(current_chunk),
                "headings": list(current_headings),
                "content": current_chunk,
            })
            chunk_index += 1
            # Compute overlap tail
            overlap_prefix = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = (overlap_prefix + "\n\n" + para).strip()
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk:
        chunks.append({
            "chunk_id": chunk_index,
            "character_count": len(current_chunk),
            "headings": list(current_headings),
            "content": current_chunk,
        })

    result = {
        "status": "success",
        "total_characters": len(content),
        "total_chunks": len(chunks),
        "max_chunk_size": max_chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks": chunks,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, mode="w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=2)

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Semantic document chunking engine preserving heading boundaries for RAG."
    )
    parser.add_argument("--input", "-i", help="Path to input text or markdown document")
    parser.add_argument("--text", "-t", help="Raw input string")
    parser.add_argument("--max-size", "-m", type=int, default=800, help="Max characters per chunk")
    parser.add_argument("--overlap", "-o", type=int, default=100, help="Overlap characters between chunks")
    parser.add_argument("--output", help="Path to write JSON chunk records")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        res = execute_semantic_chunking_engine(
            input_path=args.input,
            text=args.text,
            max_chunk_size=args.max_size,
            chunk_overlap=args.overlap,
            output_path=args.output,
        )
        print(f"Segmented {res['total_characters']} characters into {res['total_chunks']} chunks")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
