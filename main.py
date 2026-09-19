"""
Main CLI entry point for the sequential concept summarization agent.
Provides support for checkpointing, interruption, automatic resumption,
validation, and state inspection.
"""

import os
import sys
import argparse
import json
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress harmless requests dependency warning on Windows
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load environment variables from .env if present
load_dotenv()

from src.agent.state import create_initial_state, ORDERED_ITEMS, ORDERED_ITEM_IDS
from src.agent.graph import build_summarization_graph
from src.persistence.progress import (
    get_sqlite_checkpointer,
    read_progress_json,
    save_progress_json,
    corrupt_item_result,
    reset_all_state,
    PROGRESS_JSON_PATH,
    CHECKPOINTS_DB_PATH,
)
from src.llm.client import get_llm
from src.validation.validator import run_llm_validation, format_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assignment 2: Sequential Technical Concept Summarizer with LangGraph Checkpointing & Resumption."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear all persisted checkpoints and progress state before running.",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        metavar="N",
        default=None,
        help="Intentionally stop execution after processing and persisting item N (e.g. 3).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution from the last persisted checkpoint.",
    )
    parser.add_argument(
        "--corrupt",
        type=str,
        nargs="?",
        const="item3",
        default=None,
        metavar="ITEM_ID",
        help="Intentionally corrupt an item's stored summary (default: item3) for validation failure demo.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run final validation directly on current persisted results.",
    )
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="Display the current human-readable persisted state from state/progress.json.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic Mock LLM provider (no API key required).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI model name (default: gpt-4o-mini or from MODEL_NAME env).",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="LangGraph thread ID (default: assignment2-demo or from LANGGRAPH_THREAD_ID env).",
    )
    return parser.parse_args()


def display_state() -> None:
    """Print the contents of state/progress.json."""
    data = read_progress_json()
    if not data:
        print("\n[State] No persisted progress.json found in state/ directory.\n")
        return
    print("\n" + "=" * 60)
    print("CURRENT PERSISTED PROGRESS (state/progress.json)")
    print("=" * 60)
    print(json.dumps(data, indent=2))
    print("=" * 60 + "\n")


def main() -> None:
    args = parse_args()

    thread_id = args.thread_id or os.getenv("LANGGRAPH_THREAD_ID", "assignment2-demo")
    config = {"configurable": {"thread_id": thread_id}}

    # Handle --reset
    if args.reset:
        print("\n[Action] Resetting all state and checkpoints...")
        reset_all_state()
        print("[Action] Reset complete. Fresh state initialized.\n")
        if args.stop_after is None and not args.validate and not args.show_state and not args.corrupt and not args.resume:
            return

    # Handle --show-state
    if args.show_state:
        display_state()
        return

    # Handle --corrupt
    if args.corrupt:
        item_to_corrupt = args.corrupt
        bad_text = "This item is about cooking recipes and has nothing to do with the requested technical concept."
        success = corrupt_item_result(item_id=item_to_corrupt, bad_result=bad_text)
        if success:
            print(f"\n[Corruption] Successfully injected bad result into {item_to_corrupt}.")
            print(f"[Corruption] Injected text: \"{bad_text}\"\n")
        if not args.validate:
            return

    # Check existing state to determine cumulative call count
    checkpointer = get_sqlite_checkpointer()
    
    # Peek existing progress or checkpoint to know prior calls
    persisted_info = read_progress_json()
    prior_calls = persisted_info.get("total_llm_calls", 0) if persisted_info else 0

    # Initialize LLM
    try:
        llm = get_llm(model_name=args.model, use_mock=args.mock, initial_call_count=prior_calls)
    except ValueError as e:
        print(f"\n[Configuration Error] {e}\n")
        sys.exit(1)

    # Handle standalone --validate
    if args.validate:
        persisted = read_progress_json()
        if not persisted or "items" not in persisted:
            print("\n[Error] No persisted state available to validate. Run the agent first.\n")
            sys.exit(1)
        print("\n" + "=" * 60)
        print("RUNNING FINAL VALIDATION ON PERSISTED RESULTS")
        print("=" * 60)
        val_result = run_llm_validation(persisted["items"], llm)
        report = format_validation_report(val_result, persisted["items"])
        print(report)
        print("=" * 60)
        print(f"Total LLM calls: {llm.get_call_count()}")
        print("=" * 60 + "\n")
        return

    # Build LangGraph app
    app = build_summarization_graph(llm=llm, checkpointer=checkpointer)

    # Inspect current checkpoint state
    current_state_snapshot = app.get_state(config)
    has_existing_state = bool(current_state_snapshot and current_state_snapshot.values)

    if has_existing_state:
        saved_items = current_state_snapshot.values.get("items", {})
        completed_items = [
            item_id for item_id in ORDERED_ITEM_IDS
            if saved_items.get(item_id, {}).get("status") == "completed"
        ]
        
        print("\n" + "=" * 60)
        print("EXISTING CHECKPOINT DETECTED (Resuming workflow)")
        print(f"Thread ID: {thread_id}")
        print("=" * 60)
        
        # Log skipped items
        for item_id in ORDERED_ITEM_IDS:
            if item_id in completed_items:
                print(f"{item_id} -> SKIP (already completed)")
            else:
                print(f"{item_id} -> PENDING (will be processed)")
        print("=" * 60)

        # Update stop_after in state if specified in CLI
        input_state = {
            "stop_after": args.stop_after,
            "should_stop": False,
        }
        # Resume LangGraph execution from checkpoint
        app.invoke(input_state, config=config)

    else:
        print("\n" + "=" * 60)
        print("STARTING FRESH RUN (No prior checkpoint found)")
        print(f"Thread ID: {thread_id}")
        if args.stop_after:
            print(f"Deterministic Stop Configured: Stop after item {args.stop_after}")
        print("=" * 60)

        initial_state = create_initial_state(run_id=thread_id, stop_after=args.stop_after)
        app.invoke(initial_state, config=config)


if __name__ == "__main__":
    main()
