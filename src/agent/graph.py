"""
LangGraph workflow definition for sequential concept summarization,
checkpoint persistence, interruption handling, and final validation.
"""

import datetime
from typing import Dict, Any, Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState, ORDERED_ITEMS, ORDERED_ITEM_IDS
from src.agent.prompts import SUMMARIZE_SYSTEM_PROMPT, SUMMARIZE_USER_PROMPT
from src.llm.client import TrackedLLM
from src.persistence.progress import save_progress_json, read_progress_json
from src.validation.validator import run_llm_validation, format_validation_report


def build_summarization_graph(llm: TrackedLLM, checkpointer: Any = None):
    """
    Construct and compile the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # ------------------------------------------------------------------
    # Node: find_next_item
    # ------------------------------------------------------------------
    def find_next_item_node(state: AgentState) -> Dict[str, Any]:
        items = state.get("items", {})
        
        # Log skipped items if resuming
        pending_item_id = None
        for item_id in ORDERED_ITEM_IDS:
            item_data = items.get(item_id, {})
            if item_data.get("status") == "completed":
                # Only log skipped during initial scan if state indicates a resumed run
                pass
            elif item_data.get("status") == "pending" and pending_item_id is None:
                pending_item_id = item_id

        return {
            "current_item_id": pending_item_id,
        }

    # ------------------------------------------------------------------
    # Node: process_item
    # ------------------------------------------------------------------
    def process_item_node(state: AgentState) -> Dict[str, Any]:
        item_id = state.get("current_item_id")
        if not item_id:
            return {}

        items = dict(state.get("items", {}))
        item_data = dict(items.get(item_id, {}))
        topic = item_data.get("topic", "")

        print(f"\n[Process] Processing {item_id}: '{topic}'...")
        
        # Build prompt & invoke LLM
        messages = [
            SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT),
            HumanMessage(content=SUMMARIZE_USER_PROMPT.format(topic=topic)),
        ]
        response = llm.invoke(messages)
        result_text = response.content if isinstance(response.content, str) else str(response.content)

        if not result_text or not result_text.strip():
            raise ValueError(f"Received empty summary from LLM for topic: {topic}")

        # Update item state
        item_data["status"] = "completed"
        item_data["result"] = result_text.strip()
        item_data["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        item_data["llm_calls"] = item_data.get("llm_calls", 0) + 1
        items[item_id] = item_data

        total_calls = llm.get_call_count()
        completed_count = sum(1 for it in items.values() if it.get("status") == "completed")
        stop_after = state.get("stop_after")
        should_stop = bool(stop_after and completed_count >= stop_after and completed_count < len(ORDERED_ITEMS))

        updated_state: Dict[str, Any] = {
            "items": items,
            "last_completed_item": item_id,
            "total_llm_calls": total_calls,
            "should_stop": should_stop,
        }

        # Persist human-readable JSON immediately
        temp_state_to_save = dict(state)
        temp_state_to_save.update(updated_state)
        save_progress_json(temp_state_to_save)
        print(f"[Persist] Successfully persisted progress for {item_id} -> status/progress.json & checkpoint.")

        return updated_state

    # ------------------------------------------------------------------
    # Node: handle_interruption
    # ------------------------------------------------------------------
    def handle_interruption_node(state: AgentState) -> Dict[str, Any]:
        last_item = state.get("last_completed_item", "unknown")
        print("\n" + "=" * 60)
        print("INTENTIONAL INTERRUPTION")
        print(f"Progress safely persisted after {last_item}.")
        print("Run the same command without --stop-after to resume.")
        print("=" * 60 + "\n")
        return {"should_stop": True}

    # ------------------------------------------------------------------
    # Node: final_validation
    # ------------------------------------------------------------------
    def final_validation_node(state: AgentState) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("FINAL VALIDATION PHASE")
        print("Re-reading persisted results and validating against expected concepts...")
        print("=" * 60)

        # Inspect current persisted progress.json to catch any external/injected corruption
        persisted = read_progress_json()
        items = persisted.get("items", state.get("items", {})) if persisted else state.get("items", {})

        # Run single LLM validation
        val_result = run_llm_validation(items, llm)
        
        updated_state = {
            "items": items,
            "validation_complete": True,
            "validation_results": val_result,
            "total_llm_calls": llm.get_call_count(),
        }

        temp_state_to_save = dict(state)
        temp_state_to_save.update(updated_state)
        save_progress_json(temp_state_to_save)

        return updated_state

    # ------------------------------------------------------------------
    # Node: report_results
    # ------------------------------------------------------------------
    def report_results_node(state: AgentState) -> Dict[str, Any]:
        items = state.get("items", {})
        val_result = state.get("validation_results", {})
        
        # Display formatted validation report
        report_text = format_validation_report(val_result, items)
        print(report_text)

        total_calls = llm.get_call_count()
        print("=" * 60)
        print(f"Total LLM calls: {total_calls}")
        print("=" * 60 + "\n")

        return {}

    # ------------------------------------------------------------------
    # Conditional Edges
    # ------------------------------------------------------------------
    def route_after_find(state: AgentState) -> Literal["process_item", "final_validation"]:
        if state.get("current_item_id"):
            return "process_item"
        return "final_validation"

    def route_after_process(state: AgentState) -> Literal["handle_interruption", "find_next_item"]:
        if state.get("should_stop"):
            return "handle_interruption"
        return "find_next_item"

    # ------------------------------------------------------------------
    # Graph Wiring
    # ------------------------------------------------------------------
    workflow.add_node("find_next_item", find_next_item_node)
    workflow.add_node("process_item", process_item_node)
    workflow.add_node("handle_interruption", handle_interruption_node)
    workflow.add_node("final_validation", final_validation_node)
    workflow.add_node("report_results", report_results_node)

    workflow.add_edge(START, "find_next_item")
    workflow.add_conditional_edges(
        "find_next_item",
        route_after_find,
        {
            "process_item": "process_item",
            "final_validation": "final_validation",
        }
    )
    workflow.add_conditional_edges(
        "process_item",
        route_after_process,
        {
            "handle_interruption": "handle_interruption",
            "find_next_item": "find_next_item",
        }
    )
    workflow.add_edge("handle_interruption", END)
    workflow.add_edge("final_validation", "report_results")
    workflow.add_edge("report_results", END)

    return workflow.compile(checkpointer=checkpointer)
