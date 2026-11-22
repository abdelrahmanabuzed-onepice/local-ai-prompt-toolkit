"""Local AI & Prompt Engineering Toolkit.

Offline utilities for local LLM benchmarking, prompt spintax synthesis,
token estimation, semantic retrieval, and RAG chunking.
"""

__version__ = "1.0.0"
__author__ = "Abdelrahman Abouzed"

from local_ai_prompt_toolkit.tools import (
    execute_ollama_model_benchmarker,
    execute_smart_prompt_spintax_generator,
    execute_token_budget_estimator,
    execute_semantic_chunking_engine,
    execute_embedding_similarity_ranker,
    execute_prompt_template_vault,
    execute_toxic_language_filter,
    execute_ollama_vulkan_accelerator_check,
    execute_text_summarizer_extractor,
    execute_conversation_context_trimmer,
)

__all__ = [
    "__version__",
    "__author__",
    "execute_ollama_model_benchmarker",
    "execute_smart_prompt_spintax_generator",
    "execute_token_budget_estimator",
    "execute_semantic_chunking_engine",
    "execute_embedding_similarity_ranker",
    "execute_prompt_template_vault",
    "execute_toxic_language_filter",
    "execute_ollama_vulkan_accelerator_check",
    "execute_text_summarizer_extractor",
    "execute_conversation_context_trimmer",
]
