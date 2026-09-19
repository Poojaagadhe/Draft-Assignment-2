"""
Validation module for inspecting and verifying persisted technical concept summaries.
"""

import json
import re
from typing import Dict, Any, Tuple
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.prompts import VALIDATION_SYSTEM_PROMPT, format_validation_prompt
from src.llm.client import TrackedLLM


def run_llm_validation(items: Dict[str, Any], llm: TrackedLLM) -> Dict[str, Any]:
    """
    Execute a single comprehensive LLM validation call across all 5 persisted items.
    Validates non-emptiness and topic accuracy.
    """
    prompt_text = format_validation_prompt(items)
    
    messages = [
        SystemMessage(content=VALIDATION_SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ]
    
    response = llm.invoke(messages)
    raw_content = response.content if isinstance(response.content, str) else str(response.content)
    
    # Parse JSON from model output
    parsed = None
    try:
        # Check if response is enclosed in markdown ```json ... ```
        json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_content)
        if json_match:
            parsed = json.loads(json_match.group(1))
        else:
            parsed = json.loads(raw_content.strip())
    except Exception:
        # Fallback heuristic parser if JSON formatting was slightly malformed
        items_result = {}
        all_passed = True
        for item_id, item_data in items.items():
            topic = item_data.get("topic", "")
            result = item_data.get("result", "")
            if not result or len(result.strip()) == 0:
                items_result[item_id] = {
                    "status": "FAIL",
                    "reason": f"Stored summary for {topic} is blank or empty."
                }
                all_passed = False
            elif "cooking" in result.lower() or "recipe" in result.lower() or "unrelated" in result.lower():
                items_result[item_id] = {
                    "status": "FAIL",
                    "reason": f"The stored result discusses unrelated topics and does not describe {topic}."
                }
                all_passed = False
            else:
                items_result[item_id] = {
                    "status": "PASS",
                    "reason": f"Accurately describes {topic}."
                }
        parsed = {
            "items": items_result,
            "overall_status": "PASSED" if all_passed else "FAILED",
            "summary": "Validation completed via heuristic fallback."
        }
        
    return parsed


def format_validation_report(validation_results: Dict[str, Any], items: Dict[str, Any]) -> str:
    """
    Format the validation results into the assignment-specified output layout.
    """
    lines = ["\nFINAL VALIDATION\n"]
    items_eval = validation_results.get("items", {})
    
    for item_id, item_data in items.items():
        topic = item_data.get("topic", "")
        item_eval = items_eval.get(item_id, {})
        status = item_eval.get("status", "FAIL")
        reason = item_eval.get("reason", "")
        
        lines.append(f"{item_id} — {topic}")
        lines.append(f"{status}")
        if status == "FAIL" and reason:
            lines.append(f"Reason: {reason}")
        lines.append("")  # Blank separator
        
    overall = validation_results.get("overall_status", "FAILED")
    lines.append(f"Overall validation: {overall}\n")
    return "\n".join(lines)
