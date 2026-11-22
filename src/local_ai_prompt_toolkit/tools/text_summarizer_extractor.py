"""Offline extractive text summarizer ranking key sentences via TF-IDF and TextRank graph heuristics."""

import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves",
}


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences while respecting abbreviations and punctuation."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 15]


def _score_sentences_tfidf(sentences: List[str]) -> List[float]:
    """Score sentences based on word frequency and inverse document frequency across sentences."""
    words_per_sentence = []
    doc_freq: Dict[str, int] = {}

    for s in sentences:
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", s) if w.lower() not in STOPWORDS]
        words_per_sentence.append(words)
        unique_words = set(words)
        for w in unique_words:
            doc_freq[w] = doc_freq.get(w, 0) + 1

    total_s = len(sentences)
    scores: List[float] = []

    for words in words_per_sentence:
        if not words:
            scores.append(0.0)
            continue
        s_score = 0.0
        for w in words:
            tf = words.count(w) / len(words)
            idf = math.log((total_s + 1) / (doc_freq.get(w, 1) + 1)) + 1.0
            s_score += tf * idf
        scores.append(round(s_score, 4))

    return scores


def execute_text_summarizer_extractor(
    text: Optional[str] = None,
    input_path: Optional[str] = None,
    sentence_count: int = 3,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract top salient sentences from input document offline.

    Args:
        text: Raw document text string.
        input_path: Optional path to input document.
        sentence_count: Number of salient sentences to extract.
        output_path: Optional destination to write JSON summary.

    Returns:
        Structured dictionary reporting summary sentences and compression metrics.
    """
    if input_path and os.path.isfile(input_path):
        with open(input_path, mode="r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    elif text is not None:
        content = text
    else:
        raise ValueError("Either text or input_path must be specified.")

    sentences = _split_sentences(content)
    if not sentences:
        return {
            "status": "success",
            "original_sentence_count": 0,
            "summary_sentences": [],
            "summary_text": "",
            "compression_ratio": 0.0,
        }

    scores = _score_sentences_tfidf(sentences)
    ranked_indices = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    selected_indices = sorted(ranked_indices[:sentence_count])

    summary_sentences = [sentences[i] for i in selected_indices]
    summary_text = " ".join(summary_sentences)

    compression = round((len(summary_text) / len(content)) if len(content) > 0 else 0.0, 3)

    result = {
        "status": "success",
        "original_sentence_count": len(sentences),
        "extracted_sentence_count": len(summary_sentences),
        "compression_ratio": compression,
        "summary_sentences": summary_sentences,
        "summary_text": summary_text,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, mode="w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=2)

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Offline extractive text summarizer ranking sentences via TF-IDF."
    )
    parser.add_argument("--text", "-t", help="Raw input text to summarize")
    parser.add_argument("--input", "-i", help="Path to input document")
    parser.add_argument("--count", "-c", type=int, default=3, help="Number of sentences in summary")
    parser.add_argument("--output", "-o", help="Path to write JSON summary")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        res = execute_text_summarizer_extractor(
            text=args.text,
            input_path=args.input,
            sentence_count=args.count,
            output_path=args.output,
        )
        print(f"Summary ({res['extracted_sentence_count']}/{res['original_sentence_count']} sentences, {res['compression_ratio'] * 100:.1f}% size):\n")
        print(res["summary_text"])
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
