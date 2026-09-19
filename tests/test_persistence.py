"""
Tests for state persistence, atomic writes, and interruption safety.
"""

import json
import pytest
from pathlib import Path
from src.agent.state import create_initial_state, ORDERED_ITEM_IDS
from src.agent.graph import build_summarization_graph
from src.llm.client import get_llm
from src.persistence.progress import (
    save_progress_json,
    read_progress_json,
    atomic_write_json,
    reset_all_state,
    get_sqlite_checkpointer,
)


def test_atomic_write_and_read(tmp_path: Path):
    """Test that atomic writes produce valid JSON without corruption."""
    test_file = tmp_path / "progress.json"
    data = {"run_id": "test-run", "test_key": "test_value"}
    
    atomic_write_json(test_file, data)
    assert test_file.exists()
    
    loaded = read_progress_json(test_file)
    assert loaded is not None
    assert loaded["run_id"] == "test-run"
    assert loaded["test_key"] == "test_value"


def test_persistence_after_interruption(tmp_path: Path):
    """
    Test 2 & Test 8: Verify that an interruption after item3 leaves valid persisted state
    where item1..3 are completed and item4..5 are pending.
    """
    db_file = tmp_path / "checkpoints.db"
    json_file = tmp_path / "progress.json"
    
    checkpointer = get_sqlite_checkpointer(db_file)
    llm = get_llm(use_mock=True)
    app = build_summarization_graph(llm=llm, checkpointer=checkpointer)
    
    thread_id = "test-interrupt"
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = create_initial_state(run_id=thread_id, stop_after=3)
    
    # Run graph with stop_after=3
    app.invoke(initial_state, config=config)
    
    # Verify LangGraph checkpoint
    snapshot = app.get_state(config)
    assert snapshot is not None
    items_state = snapshot.values["items"]
    
    assert items_state["item1"]["status"] == "completed"
    assert items_state["item2"]["status"] == "completed"
    assert items_state["item3"]["status"] == "completed"
    assert items_state["item4"]["status"] == "pending"
    assert items_state["item5"]["status"] == "pending"
    
    assert items_state["item1"]["result"] is not None
    assert items_state["item2"]["result"] is not None
    assert items_state["item3"]["result"] is not None
    assert items_state["item4"]["result"] is None
    assert items_state["item5"]["result"] is None
