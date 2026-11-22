# Local AI Prompt Toolkit

A modular Python toolkit for local LLM inference benchmarking, polymorphic prompt engineering, semantic retrieval ranking, and RAG preprocessing pipelines.

## Architecture

```text
[ Local LLMs & Documents ]
  │   ├── Ollama Server Daemons (Llama 3, Mistral, Qwen)
  │   ├── Raw Prompts & Spintax Templates
  │   └── Large Knowledge Documents & Transcripts
  ▼
[ Prompt & Context Engine ]
  │   ├── Ollama Speed & Latency Benchmarking
  │   ├── Spintax Synthesis with Jaccard Diversity Scoring
  │   ├── Architecture-Aware Token Budget Estimation
  │   ├── Structure-Preserving Semantic Chunking for RAG
  │   ├── Offline Cosine Similarity Vector Retrieval
  │   ├── Versioned Prompt Template Vault & Parameterization
  │   ├── Toxicity & Confidential Information Leak Detection
  │   ├── Vulkan / CUDA Hardware Accelerator Diagnostics
  │   ├── Extractive Sentence Summarization via TF-IDF
  │   └── Multi-Turn Conversation History Compactor
  ▼
[ Pipeline Outputs ]
  ├── Optimized RAG Chunks & Retrieval Rankings
  ├── Hardware Performance Telemetry & Speed Benchmarks
  └── Sanitized, Budget-Compliant Execution Contexts
```

## Tool Inventory

| Tool | Module | Description |
|---|---|---|
| Ollama Benchmarker | `ollama_model_benchmarker.py` | Measures local LLM token generation throughput and prompt latency |
| Spintax Generator | `smart_prompt_spintax_generator.py` | Expands nested spintax patterns with pairwise Jaccard diversity scoring |
| Token Budget Estimator | `token_budget_estimator.py` | Calculates token requirements against context window limits |
| Semantic Chunker | `semantic_chunking_engine.py` | Segments markdown and text documents preserving headings for RAG |
| Similarity Ranker | `embedding_similarity_ranker.py` | Evaluates cosine similarity between vectors or text queries offline |
| Prompt Template Vault | `prompt_template_vault.py` | Manages versioned prompt templates and parameter interpolation |
| Toxic Language Filter | `toxic_language_filter.py` | Detects offensive speech and confidential credentials in prompts |
| Vulkan Accelerator Check | `ollama_vulkan_accelerator_check.py` | Diagnoses GPU hardware, Vulkan runtimes, and Ollama offloading |
| Text Summarizer | `text_summarizer_extractor.py` | Extracts top salient sentences using TF-IDF heuristic ranking |
| Context Trimmer | `conversation_context_trimmer.py` | Trims multi-turn conversation histories within token boundaries |

## Quickstart

### Installation

```bash
git clone https://github.com/abdelrahmanabuzed-onepice/local-ai-prompt-toolkit.git
cd local-ai-prompt-toolkit
pip install -e .
```

### Usage Examples

Run via unified CLI dispatcher:

```bash
# Display CLI help
localai-cli --help

# Generate diverse prompt variations
localai-cli spintax --template "{Write|Draft} a {clean|robust} Python script for {data|file} processing." --count 4

# Check local GPU accelerators and Vulkan drivers
localai-cli vulkan-check

# Estimate token usage for a document
localai-cli token-budget --input document.txt --model llama3-8b
```

Or execute individual tool scripts directly:

```bash
python src/local_ai_prompt_toolkit/tools/token_budget_estimator.py --text "Sample input string"
```

## Testing

Execute the unit test suite:

```bash
pytest tests/
```

## Author

Abdelrahman Abouzed  
GitHub: [abdelrahmanabuzed-onepice](https://github.com/abdelrahmanabuzed-onepice)
