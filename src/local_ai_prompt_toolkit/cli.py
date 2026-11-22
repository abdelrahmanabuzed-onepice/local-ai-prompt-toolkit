import sys
from pathlib import Path
_s = str(Path(__file__).resolve().parent.parent)
if _s not in sys.path:
    sys.path.insert(0, _s)

"""Unified Command Line Interface dispatcher for Local AI and Prompt Toolkit tools."""

import argparse
import sys
from typing import List, Optional

from local_ai_prompt_toolkit.tools.ollama_model_benchmarker import build_parser as parse_bench, main as main_bench
from local_ai_prompt_toolkit.tools.smart_prompt_spintax_generator import build_parser as parse_spintax, main as main_spintax
from local_ai_prompt_toolkit.tools.token_budget_estimator import build_parser as parse_token, main as main_token
from local_ai_prompt_toolkit.tools.semantic_chunking_engine import build_parser as parse_chunk, main as main_chunk
from local_ai_prompt_toolkit.tools.embedding_similarity_ranker import build_parser as parse_rank, main as main_rank
from local_ai_prompt_toolkit.tools.prompt_template_vault import build_parser as parse_vault, main as main_vault
from local_ai_prompt_toolkit.tools.toxic_language_filter import build_parser as parse_toxic, main as main_toxic
from local_ai_prompt_toolkit.tools.ollama_vulkan_accelerator_check import build_parser as parse_vulkan, main as main_vulkan
from local_ai_prompt_toolkit.tools.text_summarizer_extractor import build_parser as parse_summary, main as main_summary
from local_ai_prompt_toolkit.tools.conversation_context_trimmer import build_parser as parse_trim, main as main_trim


def build_parser() -> argparse.ArgumentParser:
    """Build unified multi-command argument parser."""
    parser = argparse.ArgumentParser(
        prog="localai-cli",
        description="Local AI & Prompt Toolkit: Engineering CLI suite.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available local AI subcommands")

    subparsers.add_parser("benchmark", parents=[parse_bench()], add_help=False, help="Benchmark local LLM throughput via Ollama")
    subparsers.add_parser("spintax", parents=[parse_spintax()], add_help=False, help="Synthesize prompt variations from spintax")
    subparsers.add_parser("token-budget", parents=[parse_token()], add_help=False, help="Estimate token usage across model windows")
    subparsers.add_parser("chunk", parents=[parse_chunk()], add_help=False, help="Chunk documents semantically for RAG")
    subparsers.add_parser("rank-similarity", parents=[parse_rank()], add_help=False, help="Cosine similarity ranking between vectors/text")
    subparsers.add_parser("vault", parents=[parse_vault()], add_help=False, help="Manage versioned prompt template vault")
    subparsers.add_parser("toxic-filter", parents=[parse_toxic()], add_help=False, help="Audit text for toxicity and secret leaks")
    subparsers.add_parser("vulkan-check", parents=[parse_vulkan()], add_help=False, help="Inspect GPU drivers, Vulkan, and Ollama status")
    subparsers.add_parser("summarize", parents=[parse_summary()], add_help=False, help="Extractive text summarization via TF-IDF")
    subparsers.add_parser("trim-context", parents=[parse_trim()], add_help=False, help="Trim multi-turn conversation histories")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Unified CLI dispatch routine."""
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0

    command_map = {
        "benchmark": main_bench,
        "spintax": main_spintax,
        "token-budget": main_token,
        "chunk": main_chunk,
        "rank-similarity": main_rank,
        "vault": main_vault,
        "toxic-filter": main_toxic,
        "vulkan-check": main_vulkan,
        "summarize": main_summary,
        "trim-context": main_trim,
    }

    subcommand = argv[0]
    if subcommand in command_map:
        return command_map[subcommand](argv[1:])

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
