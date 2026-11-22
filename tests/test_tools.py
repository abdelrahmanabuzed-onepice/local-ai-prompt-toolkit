"""Unit tests for all 10 local AI prompt toolkit tools."""

import json
import os
import sys
import tempfile
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_ai_prompt_toolkit.tools.ollama_model_benchmarker import execute_ollama_model_benchmarker
from local_ai_prompt_toolkit.tools.smart_prompt_spintax_generator import execute_smart_prompt_spintax_generator
from local_ai_prompt_toolkit.tools.token_budget_estimator import execute_token_budget_estimator
from local_ai_prompt_toolkit.tools.semantic_chunking_engine import execute_semantic_chunking_engine
from local_ai_prompt_toolkit.tools.embedding_similarity_ranker import execute_embedding_similarity_ranker
from local_ai_prompt_toolkit.tools.prompt_template_vault import execute_prompt_template_vault
from local_ai_prompt_toolkit.tools.toxic_language_filter import execute_toxic_language_filter
from local_ai_prompt_toolkit.tools.ollama_vulkan_accelerator_check import execute_ollama_vulkan_accelerator_check
from local_ai_prompt_toolkit.tools.text_summarizer_extractor import execute_text_summarizer_extractor
from local_ai_prompt_toolkit.tools.conversation_context_trimmer import execute_conversation_context_trimmer


def test_tool_01_ollama_benchmarker():
    # Test handling offline gracefully
    res = execute_ollama_model_benchmarker(host="http://127.0.0.1:59999")
    assert res["status"] in ("offline", "success")
    assert "tokens_per_second" in res


def test_tool_02_spintax_generator():
    template = "{Write|Compose} a {clean|robust} Python script for {data|file} validation."
    res = execute_smart_prompt_spintax_generator(template=template, count=3, seed=42)
    assert res["status"] == "success"
    assert res["generated_count"] >= 2
    assert "average_uniqueness" in res


def test_tool_03_token_budget_estimator():
    sample_prompt = "Explain the difference between TCP and UDP sockets in network programming."
    res = execute_token_budget_estimator(text=sample_prompt, target_model="llama3-8b")
    assert res["status"] == "success"
    assert res["estimated_prompt_tokens"] > 0
    assert res["fits_in_context"] is True


def test_tool_04_semantic_chunking():
    doc = "# Chapter 1: Introduction\n\nThis is paragraph one.\n\nThis is paragraph two.\n\n# Chapter 2: Methods\n\nMethod details follow."
    res = execute_semantic_chunking_engine(text=doc, max_chunk_size=100, chunk_overlap=20)
    assert res["status"] == "success"
    assert res["total_chunks"] >= 2
    assert "# Chapter 1: Introduction" in res["chunks"][0]["headings"]


def test_tool_05_similarity_ranker():
    # Test vector cosine similarity
    vec_q = [1.0, 0.0, 0.5]
    docs = [
        {"id": "doc1", "vector": [1.0, 0.0, 0.5]},
        {"id": "doc2", "vector": [0.0, 1.0, 0.0]},
    ]
    res = execute_embedding_similarity_ranker(query_vector=vec_q, document_vectors=docs)
    assert res["status"] == "success"
    assert res["results"][0]["id"] == "doc1"
    assert res["results"][0]["similarity_score"] == 1.0


def test_tool_06_prompt_vault(tmp_path):
    vault_file = tmp_path / "vault.json"
    template_str = "You are an expert in {{domain}}. Explain {{topic}} clearly."

    # Register
    res_reg = execute_prompt_template_vault(
        action="register",
        vault_path=str(vault_file),
        name="expert_explainer",
        version="1.0.0",
        template_text=template_str,
    )
    assert res_reg["status"] == "success"

    # Render
    res_render = execute_prompt_template_vault(
        action="render",
        vault_path=str(vault_file),
        name="expert_explainer",
        params={"domain": "Operating Systems", "topic": "Virtual Memory"},
    )
    assert res_render["status"] == "success"
    assert "You are an expert in Operating Systems" in res_render["rendered_prompt"]


def test_tool_07_toxic_language_filter():
    # Clean text
    clean_res = execute_toxic_language_filter(text="Deterministic and verifiable system architecture.")
    assert clean_res["is_clean"] is True

    # Secret leak detection
    leaked_res = execute_toxic_language_filter(
        text="Debug error: api_key = 'sk-1234567890abcdef123456'",
        redact=True,
    )
    assert leaked_res["is_clean"] is False
    assert leaked_res["sanitized_text"] is not None
    assert "REDACTED" in leaked_res["sanitized_text"]


def test_tool_08_vulkan_accelerator_check():
    res = execute_ollama_vulkan_accelerator_check(host="http://127.0.0.1:59999")
    assert res["status"] == "success"
    assert "vulkan_runtime_detected" in res
    assert "gpus_detected" in res


def test_tool_09_text_summarizer():
    doc = (
        "Distributed consensus algorithms ensure state consistency across failing nodes. "
        "The Raft consensus algorithm is designed to be easily understandable. "
        "Raft decomposes consensus into leader election, log replication, and safety. "
        "In contrast, Paxos is notoriously difficult to comprehend and implement in production. "
        "Practical engineering teams frequently choose Raft for distributed metadata stores."
    )
    res = execute_text_summarizer_extractor(text=doc, sentence_count=2)
    assert res["status"] == "success"
    assert res["extracted_sentence_count"] == 2
    assert len(res["summary_text"]) < len(doc)


def test_tool_10_context_trimmer():
    messages = [
        {"role": "system", "content": "You are a concise engineering assistant."},
        {"role": "user", "content": "What is eBPF?"},
        {"role": "assistant", "content": "Extended Berkeley Packet Filter runs sandboxed programs in the Linux kernel."},
        {"role": "user", "content": "How does it achieve safety?"},
        {"role": "assistant", "content": "An in-kernel verifier validates that the bytecode terminates safely without memory faults."},
    ]
    res = execute_conversation_context_trimmer(
        messages=messages,
        max_tokens=40,
        keep_system_prompt=True,
        preserve_last_n_turns=2,
    )
    assert res["status"] == "success"
    assert res["messages"][0]["role"] == "system"
    assert len(res["messages"]) <= len(messages)
