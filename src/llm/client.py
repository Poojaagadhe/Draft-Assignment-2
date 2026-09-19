"""
LLM Client wrapper with explicit invocation tracking, supporting Google Gemini, OpenAI, and Mock providers.
"""

import os
import json
import re
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration


# Deterministic high-quality responses for the 5 concepts
MOCK_CONCEPT_RESPONSES = {
    "Redis Caching": (
        "Redis caching is an in-memory key-value data store used as a high-speed caching layer. "
        "It is used to dramatically reduce database read latency, offload repetitive queries, "
        "and handle high-throughput read traffic by serving hot data directly from RAM. "
        "For example, an e-commerce platform caches frequently accessed product catalog pages and user sessions in Redis. "
        "One important tradeoff is volatility and memory cost: RAM is significantly more expensive than persistent disk storage, "
        "and cache invalidation strategies (such as TTL or write-through) must be carefully engineered to prevent serving stale data."
    ),
    "Database Indexing": (
        "Database indexing is a data structure technique (typically utilizing B-Trees or Hash tables) "
        "that optimizes the speed of data retrieval operations on database tables. "
        "It is used to prevent costly full-table scans by allowing the query engine to locate target rows in logarithmic time. "
        "For example, creating a composite index on (user_id, created_at) enables instantaneous retrieval of a user's recent orders. "
        "One important tradeoff is write overhead and storage consumption: every insert, update, and delete operation "
        "must update both the underlying table and all associated indexes, increasing write latency and storage requirements."
    ),
    "Message Queues": (
        "Message queues are asynchronous communication mechanisms (such as RabbitMQ or Apache Kafka) "
        "that decouple producers and consumers via ordered message buffers. "
        "They are used to facilitate asynchronous processing, smooth out traffic spikes (load leveling), and increase overall system resilience. "
        "For example, a payment gateway places video encoding or receipt generation tasks into a queue for background worker consumption. "
        "One important tradeoff is operational complexity: introducing message queues adds challenges such as handling duplicate messages, "
        "guaranteeing delivery semantics, and managing dead-letter queues."
    ),
    "Horizontal Scaling": (
        "Horizontal scaling (scaling out) is the practice of expanding system capacity by adding more compute instances "
        "or nodes to a distributed cluster rather than upgrading a single machine. "
        "It is used to achieve virtually limitless scalability, improve fault tolerance, and avoid single-point-of-failure bottlenecks. "
        "For example, a web application provisions 20 stateless container instances behind an auto-scaler during high-traffic events. "
        "One important tradeoff is distributed system complexity: applications must be designed to be stateless, requiring centralized "
        "session stores, distributed tracing, and coordination protocols across multiple nodes."
    ),
    "Load Balancing": (
        "Load balancing is the technique of distributing incoming network traffic across multiple backend servers or resource pools. "
        "It is used to maximize throughput, minimize response latency, prevent any single server from becoming overloaded, and provide seamless failover. "
        "For example, an NGINX or AWS ALB instance distributes HTTPS user requests across a pool of application servers using round-robin or least-connections routing. "
        "One important tradeoff is the load balancer itself can become a single point of failure or network bottleneck "
        "unless configured with redundant, active-passive or multi-region high-availability pairs."
    )
}


class MockLLM(BaseChatModel):
    """
    Deterministic Mock LLM for testing and offline execution.
    Generates realistic, structured concept summaries and performs intelligent validation.
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        full_text = " ".join([m.content for m in messages if isinstance(m.content, str)])
        
        # Check if this is a validation prompt
        if "validate the following stored summaries" in full_text.lower() or "strict technical verification" in full_text.lower():
            response_content = self._perform_validation(full_text)
        else:
            # Check for concept summarization
            response_content = None
            for topic, summary in MOCK_CONCEPT_RESPONSES.items():
                if topic.lower() in full_text.lower():
                    response_content = summary
                    break
            
            if not response_content:
                response_content = (
                    "This is a standard technical summary explaining the requested architecture concept, "
                    "its core benefits, concrete operational examples, and primary architectural tradeoffs."
                )

        generation = ChatGeneration(message=AIMessage(content=response_content))
        return ChatResult(generations=[generation])

    def _perform_validation(self, prompt_text: str) -> str:
        """Inspect stored summaries and produce valid JSON evaluation."""
        items_eval = {}
        all_passed = True
        
        # Extraction of stored summaries from prompt
        item_blocks = re.findall(r"--- ITEM:\s*(item\d+)\s*---\s*Expected Topic:\s*([^\n]+)\s*Stored Summary:\s*\"\"\"([\s\S]*?)\"\"\"", prompt_text)
        
        if not item_blocks:
            for item_id, topic in [("item1", "Redis Caching"), ("item2", "Database Indexing"), ("item3", "Message Queues"), ("item4", "Horizontal Scaling"), ("item5", "Load Balancing")]:
                items_eval[item_id] = {
                    "status": "PASS",
                    "reason": f"Accurately describes {topic}."
                }
            return json.dumps({
                "items": items_eval,
                "overall_status": "PASSED",
                "summary": "All 5 concepts verified successfully."
            }, indent=2)

        for item_id, topic, stored_summary in item_blocks:
            topic = topic.strip()
            stored_summary = stored_summary.strip()
            
            # Check if summary is blank
            if not stored_summary or stored_summary == "[EMPTY]":
                items_eval[item_id] = {
                    "status": "FAIL",
                    "reason": f"Stored summary for {topic} is empty."
                }
                all_passed = False
            # Check if summary is off-topic (e.g. cooking, recipe, unrelated text)
            elif any(bad_word in stored_summary.lower() for bad_word in ["recipe", "cooking", "unrelated", "cake", "baking", "nothing to do with"]):
                items_eval[item_id] = {
                    "status": "FAIL",
                    "reason": f"The stored result discusses unrelated topics (cooking/recipes) instead of {topic}."
                }
                all_passed = False
            elif len(stored_summary) < 20:
                items_eval[item_id] = {
                    "status": "FAIL",
                    "reason": f"Summary is too short or invalid for {topic}."
                }
                all_passed = False
            else:
                items_eval[item_id] = {
                    "status": "PASS",
                    "reason": f"Accurately and comprehensively describes {topic}."
                }

        overall_status = "PASSED" if all_passed else "FAILED"
        return json.dumps({
            "items": items_eval,
            "overall_status": overall_status,
            "summary": "Validation completed successfully." if all_passed else "One or more item summaries failed validation."
        }, indent=2)

    @property
    def _llm_type(self) -> str:
        return "mock_chat_model"


class TrackedLLM:
    """
    Central wrapper around any ChatModel that explicitly counts invocations.
    Guarantees accurate LLM call accounting across multiple graph executions.
    """

    def __init__(self, model: BaseChatModel, initial_call_count: int = 0):
        self.model = model
        self.call_count = initial_call_count

    def invoke(self, messages: Any, **kwargs: Any) -> BaseMessage:
        """Invoke the underlying LLM and increment call count."""
        self.call_count += 1
        return self.model.invoke(messages, **kwargs)

    def get_call_count(self) -> int:
        return self.call_count

    def reset_call_count(self) -> None:
        self.call_count = 0


def get_llm(
    model_name: Optional[str] = None,
    use_mock: bool = False,
    temperature: float = 0.0,
    initial_call_count: int = 0,
) -> TrackedLLM:
    """
    Factory function to initialize a TrackedLLM instance.
    Supports Google Gemini (GOOGLE_API_KEY / GEMINI_API_KEY),
    OpenAI (OPENAI_API_KEY), and MockLLM.
    """
    env_mock = os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes")
    
    if use_mock or env_mock:
        return TrackedLLM(MockLLM(), initial_call_count=initial_call_count)

    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        selected_model = model_name or os.getenv("MODEL_NAME", "gemini-1.5-flash")
        model = ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=google_key,
            temperature=temperature,
        )
        return TrackedLLM(model, initial_call_count=initial_call_count)

    if openai_key:
        from langchain_openai import ChatOpenAI
        selected_model = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        model = ChatOpenAI(
            model=selected_model,
            temperature=temperature,
            api_key=openai_key,
        )
        return TrackedLLM(model, initial_call_count=initial_call_count)

    raise ValueError(
        "No API key found.\n"
        "Please provide GOOGLE_API_KEY (or OPENAI_API_KEY) in your .env file,\n"
        "or run with --mock for deterministic simulation without API keys."
    )
