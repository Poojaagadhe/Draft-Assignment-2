"""
Tests for LLM-based validation: detecting blank results, corrupted results, and valid results.
"""

import pytest
from src.validation.validator import run_llm_validation, format_validation_report
from src.llm.client import get_llm


def test_validation_passes_all_valid_results():
    """Verify that 5 properly formatted technical summaries pass validation."""
    llm = get_llm(use_mock=True)
    items = {
        "item1": {"topic": "Redis Caching", "result": "Redis caching is an in-memory data store for fast reads and low latency. Tradeoff is memory cost."},
        "item2": {"topic": "Database Indexing", "result": "Database indexing creates B-Tree structures to speed up SELECT queries. Tradeoff is slower writes."},
        "item3": {"topic": "Message Queues", "result": "Message queues decouple producers and consumers for asynchronous processing. Tradeoff is operational complexity."},
        "item4": {"topic": "Horizontal Scaling", "result": "Horizontal scaling adds more instances behind a load balancer to increase capacity. Tradeoff is state management."},
        "item5": {"topic": "Load Balancing", "result": "Load balancing distributes network traffic evenly across servers. Tradeoff is load balancer high-availability requirements."},
    }
    
    val_res = run_llm_validation(items, llm)
    assert val_res["overall_status"] == "PASSED"
    for item_id in ["item1", "item2", "item3", "item4", "item5"]:
        assert val_res["items"][item_id]["status"] == "PASS"


def test_validation_catches_blank_result():
    """
    Test 5: Verify that validation detects a blank/empty summary.
    """
    llm = get_llm(use_mock=True)
    items = {
        "item1": {"topic": "Redis Caching", "result": "Valid Redis summary."},
        "item2": {"topic": "Database Indexing", "result": ""},  # Blank result
        "item3": {"topic": "Message Queues", "result": "Valid Message Queues summary."},
        "item4": {"topic": "Horizontal Scaling", "result": "Valid Horizontal Scaling summary."},
        "item5": {"topic": "Load Balancing", "result": "Valid Load Balancing summary."},
    }
    
    val_res = run_llm_validation(items, llm)
    assert val_res["overall_status"] == "FAILED"
    assert val_res["items"]["item2"]["status"] == "FAIL"
    assert "empty" in val_res["items"]["item2"]["reason"].lower() or "blank" in val_res["items"]["item2"]["reason"].lower()
    assert val_res["items"]["item1"]["status"] == "PASS"


def test_validation_catches_wrong_corrupted_result():
    """
    Test 6: Verify that validation detects an unrelated/corrupted summary (e.g., cooking recipes).
    """
    llm = get_llm(use_mock=True)
    items = {
        "item1": {"topic": "Redis Caching", "result": "Redis caching is an in-memory key-value store for high throughput."},
        "item2": {"topic": "Database Indexing", "result": "Database indexing speeds up queries using B-Tree indexing on key columns."},
        "item3": {"topic": "Message Queues", "result": "This item is about cooking recipes and has nothing to do with the requested technical concept."},
        "item4": {"topic": "Horizontal Scaling", "result": "Horizontal scaling adds more stateless nodes across the cluster."},
        "item5": {"topic": "Load Balancing", "result": "Load balancing routes incoming network requests across compute nodes."},
    }
    
    val_res = run_llm_validation(items, llm)
    assert val_res["overall_status"] == "FAILED"
    assert val_res["items"]["item3"]["status"] == "FAIL"
    assert "cooking" in val_res["items"]["item3"]["reason"].lower() or "recipe" in val_res["items"]["item3"]["reason"].lower() or "unrelated" in val_res["items"]["item3"]["reason"].lower()
    
    assert val_res["items"]["item1"]["status"] == "PASS"
    assert val_res["items"]["item2"]["status"] == "PASS"
    assert val_res["items"]["item4"]["status"] == "PASS"
    assert val_res["items"]["item5"]["status"] == "PASS"
