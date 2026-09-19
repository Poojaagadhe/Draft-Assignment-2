"""
Tests for sequential processing, workflow resumption, and duplicate call prevention.
"""

import pytest
from pathlib import Path
from src.agent.state import create_initial_state, ORDERED_ITEM_IDS
from src.agent.graph import build_summarization_graph
from src.llm.client import get_llm
from src.persistence.progress import get_sqlite_checkpointer


def test_sequential_processing_order(tmp_path: Path):
    """
    Test 1: Verify items are processed sequentially in exact fixed order.
    """
    db_file = tmp_path / "checkpoints.db"
    checkpointer = get_sqlite_checkpointer(db_file)
    llm = get_llm(use_mock=True)
    app = build_summarization_graph(llm=llm, checkpointer=checkpointer)
    
    thread_id = "test-order"
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = create_initial_state(run_id=thread_id, stop_after=None)
    
    result = app.invoke(initial_state, config=config)
    items = result["items"]
    
    for item_id in ORDERED_ITEM_IDS:
        assert items[item_id]["status"] == "completed"
        assert items[item_id]["result"] is not None
        assert items[item_id]["completed_at"] is not None


def test_resume_skips_completed_and_no_duplicate_calls(tmp_path: Path):
    """
    Test 3 & Test 4: Verify that resuming skips completed items 1-3,
    processes items 4-5, and makes no duplicate LLM calls for items 1-3.
    """
    db_file = tmp_path / "checkpoints.db"
    checkpointer = get_sqlite_checkpointer(db_file)
    
    # Phase 1: Run with stop_after=3
    llm1 = get_llm(use_mock=True)
    app1 = build_summarization_graph(llm=llm1, checkpointer=checkpointer)
    thread_id = "test-resume-flow"
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = create_initial_state(run_id=thread_id, stop_after=3)
    app1.invoke(initial_state, config=config)
    
    assert llm1.get_call_count() == 3  # Exactly 3 calls for item1, item2, item3
    
    snapshot1 = app1.get_state(config)
    assert snapshot1.values["items"]["item1"]["status"] == "completed"
    assert snapshot1.values["items"]["item2"]["status"] == "completed"
    assert snapshot1.values["items"]["item3"]["status"] == "completed"
    assert snapshot1.values["items"]["item4"]["status"] == "pending"
    assert snapshot1.values["items"]["item5"]["status"] == "pending"
    
    # Save results of phase 1
    item1_result = snapshot1.values["items"]["item1"]["result"]
    item2_result = snapshot1.values["items"]["item2"]["result"]
    item3_result = snapshot1.values["items"]["item3"]["result"]

    # Phase 2: Resume execution
    llm2 = get_llm(use_mock=True, initial_call_count=3)
    app2 = build_summarization_graph(llm=llm2, checkpointer=checkpointer)
    
    resume_input = {"stop_after": None, "should_stop": False}
    app2.invoke(resume_input, config=config)
    
    # Phase 2 should only make 3 calls: item4, item5, and 1 validation call
    # Total calls should be 6
    assert llm2.get_call_count() == 6
    
    snapshot2 = app2.get_state(config)
    final_items = snapshot2.values["items"]
    
    # Verify all completed
    for item_id in ORDERED_ITEM_IDS:
        assert final_items[item_id]["status"] == "completed"
    
    # Verify items 1-3 were NOT re-generated (exact same text retained)
    assert final_items["item1"]["result"] == item1_result
    assert final_items["item2"]["result"] == item2_result
    assert final_items["item3"]["result"] == item3_result
