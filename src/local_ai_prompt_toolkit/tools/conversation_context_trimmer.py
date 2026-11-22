"""Multi-turn conversation chat history compactor and sliding context trimmer."""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


def execute_conversation_context_trimmer(
    messages: Optional[List[Dict[str, str]]] = None,
    input_path: Optional[str] = None,
    max_tokens: int = 4000,
    keep_system_prompt: bool = True,
    preserve_last_n_turns: int = 4,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Trim multi-turn conversation messages to fit within token boundaries.

    Args:
        messages: List of message dictionaries with 'role' and 'content'.
        input_path: Path to JSON file containing messages array.
        max_tokens: Token ceiling for preserved conversation context.
        keep_system_prompt: Ensure initial system message is never pruned.
        preserve_last_n_turns: Always preserve the most recent user/assistant turns.
        output_path: Optional path to save trimmed conversation JSON.

    Returns:
        Structured dictionary reporting trimmed history and token counts.
    """
    if input_path and os.path.isfile(input_path):
        with open(input_path, mode="r", encoding="utf-8") as f:
            raw_data = json.load(f)
            messages = raw_data if isinstance(raw_data, list) else raw_data.get("messages", [])
    elif messages is None:
        raise ValueError("Either messages or input_path must be specified.")

    # Estimate tokens per message (4 chars ~= 1 token + 4 tokens overhead)
    def estimate_msg_tokens(msg: Dict[str, str]) -> int:
        return max(1, len(msg.get("content", "")) // 4) + 4

    total_initial_tokens = sum(estimate_msg_tokens(m) for m in messages)
    if total_initial_tokens <= max_tokens:
        result = {
            "status": "success",
            "trimmed": False,
            "original_message_count": len(messages),
            "final_message_count": len(messages),
            "original_estimated_tokens": total_initial_tokens,
            "final_estimated_tokens": total_initial_tokens,
            "messages": messages,
        }
    else:
        # Separate system messages and conversation turns
        system_msgs = [m for m in messages if m.get("role") == "system"] if keep_system_prompt else []
        non_system_msgs = [m for m in messages if m.get("role") != "system"] if keep_system_prompt else list(messages)

        system_tokens = sum(estimate_msg_tokens(m) for m in system_msgs)
        budget_for_turns = max_tokens - system_tokens

        # Protect last N messages
        recent_tail = non_system_msgs[-preserve_last_n_turns:] if len(non_system_msgs) >= preserve_last_n_turns else non_system_msgs
        older_head = non_system_msgs[:-preserve_last_n_turns] if len(non_system_msgs) >= preserve_last_n_turns else []

        selected_turns: List[Dict[str, str]] = list(recent_tail)
        current_turn_tokens = sum(estimate_msg_tokens(m) for m in selected_turns)

        # Prepend older messages from most recent to oldest until budget exhausted
        for msg in reversed(older_head):
            msg_tok = estimate_msg_tokens(msg)
            if current_turn_tokens + msg_tok <= budget_for_turns:
                selected_turns.insert(0, msg)
                current_turn_tokens += msg_tok
            else:
                break

        final_messages = system_msgs + selected_turns
        final_tokens = sum(estimate_msg_tokens(m) for m in final_messages)

        result = {
            "status": "success",
            "trimmed": True,
            "original_message_count": len(messages),
            "final_message_count": len(final_messages),
            "pruned_message_count": len(messages) - len(final_messages),
            "original_estimated_tokens": total_initial_tokens,
            "final_estimated_tokens": final_tokens,
            "messages": final_messages,
        }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, mode="w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=2)

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Multi-turn conversation chat history compactor and sliding context trimmer."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to JSON file containing messages")
    parser.add_argument("--max-tokens", "-m", type=int, default=4000, help="Maximum allowable tokens")
    parser.add_argument(
        "--no-system",
        dest="keep_system",
        action="store_false",
        help="Do not strictly preserve system prompt",
    )
    parser.add_argument("--preserve-turns", "-p", type=int, default=4, help="Recent turns to guarantee preservation")
    parser.add_argument("--output", "-o", help="Path to write trimmed JSON output")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        res = execute_conversation_context_trimmer(
            input_path=args.input,
            max_tokens=args.max_tokens,
            keep_system_prompt=args.keep_system,
            preserve_last_n_turns=args.preserve_turns,
            output_path=args.output,
        )
        print(f"Conversation Context: {res['original_message_count']} -> {res['final_message_count']} messages ({res['final_estimated_tokens']} tokens, trimmed: {res['trimmed']})")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
