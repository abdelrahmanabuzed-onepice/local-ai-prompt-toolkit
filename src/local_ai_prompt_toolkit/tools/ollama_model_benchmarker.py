"""Ollama local LLM performance benchmarker evaluating throughput and prompt eval latency."""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def execute_ollama_model_benchmarker(
    model: str = "llama3:latest",
    prompt: str = "Explain the difference between synchronous and asynchronous I/O in three concise sentences.",
    host: str = "http://127.0.0.1:11434",
    max_tokens: int = 100,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Benchmark local LLM inference speed, evaluation time, and token generation rate.

    Args:
        model: Model tag identifier.
        prompt: Benchmark prompt payload.
        host: Base URL of local Ollama daemon.
        max_tokens: Maximum tokens to generate.
        output_path: Optional path to write JSON benchmark report.

    Returns:
        Structured dictionary reporting tokens/sec, latency, and duration metrics.
    """
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.2,
        },
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"},
    )

    start_wall = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed_wall = time.perf_counter() - start_wall

        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)

        # Convert nanoseconds to seconds
        eval_duration_s = eval_duration_ns / 1e9 if eval_duration_ns > 0 else elapsed_wall
        prompt_eval_s = prompt_eval_duration_ns / 1e9 if prompt_eval_duration_ns > 0 else 0.0

        tokens_per_sec = round(eval_count / eval_duration_s, 2) if eval_duration_s > 0 else 0.0

        result = {
            "status": "success",
            "daemon_online": True,
            "model": model,
            "host": host,
            "prompt_tokens": prompt_eval_count,
            "generated_tokens": eval_count,
            "tokens_per_second": tokens_per_sec,
            "generation_time_seconds": round(eval_duration_s, 3),
            "prompt_eval_time_seconds": round(prompt_eval_s, 3),
            "wall_time_seconds": round(elapsed_wall, 3),
            "response_sample": data.get("response", "")[:120],
        }

    except (urllib.error.URLError, ConnectionRefusedError, OSError) as ex:
        elapsed_wall = time.perf_counter() - start_wall
        result = {
            "status": "offline",
            "daemon_online": False,
            "model": model,
            "host": host,
            "error": f"Ollama daemon unreachable at {host}: {ex}",
            "wall_time_seconds": round(elapsed_wall, 3),
            "tokens_per_second": 0.0,
        }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, mode="w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=2)

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Ollama local LLM performance benchmarker evaluating throughput and latency."
    )
    parser.add_argument("--model", "-m", default="llama3:latest", help="Ollama model identifier")
    parser.add_argument("--prompt", "-p", default="Explain synchronous vs asynchronous I/O.", help="Evaluation prompt")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama server host URL")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum generated tokens")
    parser.add_argument("--output", "-o", help="Path to write JSON benchmark report")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        res = execute_ollama_model_benchmarker(
            model=args.model,
            prompt=args.prompt,
            host=args.host,
            max_tokens=args.max_tokens,
            output_path=args.output,
        )
        if res["daemon_online"]:
            print(f"Model {res['model']}: {res['tokens_per_second']} tokens/sec ({res['generated_tokens']} tokens in {res['generation_time_seconds']}s)")
        else:
            print(f"Notice: {res['error']}")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
