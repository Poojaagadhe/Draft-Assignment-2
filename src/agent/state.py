"""
State definitions and constants for the LangGraph sequential summarization agent.
"""

from typing import Dict, List, Optional, Any, TypedDict, Literal
from pydantic import BaseModel, Field


# Canonical 5 technical concepts to process sequentially
ORDERED_ITEMS: List[Dict[str, str]] = [
    {"id": "item1", "topic": "Redis Caching"},
    {"id": "item2", "topic": "Database Indexing"},
    {"id": "item3", "topic": "Message Queues"},
    {"id": "item4", "topic": "Horizontal Scaling"},
    {"id": "item5", "topic": "Load Balancing"},
]

ORDERED_ITEM_IDS: List[str] = [item["id"] for item in ORDERED_ITEMS]
ITEM_TOPICS: Dict[str, str] = {item["id"]: item["topic"] for item in ORDERED_ITEMS}


class ItemProgress(TypedDict):
    topic: str
    status: Literal["pending", "completed", "failed"]
    result: Optional[str]
    llm_calls: int
    completed_at: Optional[str]


class ValidationItemResult(BaseModel):
    item_id: str
    topic: str
    status: Literal["PASS", "FAIL"]
    reason: str


class OverallValidationResult(BaseModel):
    items: Dict[str, ValidationItemResult]
    overall_status: Literal["PASSED", "FAILED"]
    summary: str


class AgentState(TypedDict):
    """LangGraph State representation persisted across checkpoints."""
    run_id: str
    items: Dict[str, ItemProgress]
    current_item_id: Optional[str]
    total_llm_calls: int
    stop_after: Optional[int]
    should_stop: bool
    validation_complete: bool
    validation_results: Optional[Dict[str, Any]]
    last_completed_item: Optional[str]
    error: Optional[str]


def create_initial_state(run_id: str = "assignment2-demo", stop_after: Optional[int] = None) -> AgentState:
    """Initialize a fresh agent state with all items marked as pending."""
    items_state: Dict[str, ItemProgress] = {}
    for item in ORDERED_ITEMS:
        items_state[item["id"]] = {
            "topic": item["topic"],
            "status": "pending",
            "result": None,
            "llm_calls": 0,
            "completed_at": None,
        }

    return {
        "run_id": run_id,
        "items": items_state,
        "current_item_id": None,
        "total_llm_calls": 0,
        "stop_after": stop_after,
        "should_stop": False,
        "validation_complete": False,
        "validation_results": None,
        "last_completed_item": None,
        "error": None,
    }
