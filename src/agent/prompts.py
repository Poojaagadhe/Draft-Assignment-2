"""
System and user prompts for technical summarization and validation.
"""

from typing import Dict, Any


SUMMARIZE_SYSTEM_PROMPT = """You are a senior technical writer and software architect.
Provide concise, accurate, and structured technical explanations."""

SUMMARIZE_USER_PROMPT = """Create a concise technical summary for the following concept:

Topic: {topic}

Your summary MUST contain:
1. What it is: A clear definition.
2. Why it is used: The primary architectural or engineering benefits.
3. One practical example: A concrete real-world use case.
4. One important tradeoff: A downside, cost, or operational consideration.

Target length: Approximately 80-150 words. Be direct, clear, and technical."""


VALIDATION_SYSTEM_PROMPT = """You are a strict technical verification validator.
Your role is to inspect technical concept summaries and verify if each one accurately and correctly describes its assigned topic.

For each item:
- Check that the summary is not empty or blank.
- Check that the summary specifically discusses the expected topic and is not an unrelated text (e.g. cooking, sports, generic gibberish, or a completely different technical domain).
- Mark status as "PASS" if it correctly and genuinely describes the topic.
- Mark status as "FAIL" if it is empty, off-topic, unrelated, or misleading. Provide a concise, clear reason for failure.

Return your response strictly in JSON format matching this schema:
{
  "items": {
    "item1": {"status": "PASS", "reason": "Accurately describes Redis Caching."},
    "item2": {"status": "PASS", "reason": "Accurately describes Database Indexing."},
    "item3": {"status": "FAIL", "reason": "The stored result discusses cooking recipes instead of Message Queues."},
    ...
  },
  "overall_status": "PASSED" or "FAILED",
  "summary": "Overall evaluation summary"
}
"""


def format_validation_prompt(items: Dict[str, Any]) -> str:
    """Format the validation input listing all items, expected topics, and saved results."""
    lines = ["Please validate the following stored summaries against their expected technical topics:\n"]
    for item_id, item_data in items.items():
        topic = item_data.get("topic", "")
        result = item_data.get("result", "")
        lines.append(f"--- ITEM: {item_id} ---")
        lines.append(f"Expected Topic: {topic}")
        lines.append(f"Stored Summary:\n\"\"\"{result if result else '[EMPTY]'}\"\"\"\n")

    lines.append("Evaluate each item and output the required JSON format.")
    return "\n".join(lines)
