"""
Tests for explicit LLM call counting and wrapper invocation accuracy.
"""

from pathlib import Path
from langchain_core.messages import HumanMessage
from src.llm.client import get_llm, TrackedLLM, MockLLM
from src.agent.state import create_initial_state
from src.agent.graph import build_summarization_graph
from src.persistence.progress import get_sqlite_checkpointer


def test_tracked_llm_counter_increments_only_on_invoke():
    """
    Test 7: Verify counter starts at 0 and increments strictly upon invocation.
    """
    mock_base = MockLLM()
    tracked = TrackedLLM(mock_base)
    
    assert tracked.get_call_count() == 0
    
    # Prompt construction / non-LLM actions do not increment counter
    prompt = "Summarize Redis Caching"
    assert tracked.get_call_count() == 0
    
    # First invocation
    tracked.invoke([HumanMessage(content=prompt)])
    assert tracked.get_call_count() == 1
    
    # Second invocation
    tracked.invoke([HumanMessage(content="Summarize Database Indexing")])
    assert tracked.get_call_count() == 2
    
    # Reset
    tracked.reset_call_count()
    assert tracked.get_call_count() == 0


def test_full_workflow_llm_call_count(tmp_path: Path):
    """
    Verify full run makes exactly 6 LLM calls (5 item summaries + 1 validation).
    """
    db_file = tmp_path / "checkpoints.db"
    checkpointer = get_sqlite_checkpointer(db_file)
    llm = get_llm(use_mock=True)
    app = build_summarization_graph(llm=llm, checkpointer=checkpointer)
    
    thread_id = "test-full-counter"
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = create_initial_state(run_id=thread_id, stop_after=None)
    
    app.invoke(initial_state, config=config)
    
    # 5 concept summaries + 1 validation = 6
    assert llm.get_call_count() == 6
