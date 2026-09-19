"""
Persistence module managing human-readable JSON state and SQLite checkpoints.
Provides atomic writes, state inspection, corruption simulation, and reset utilities.
"""

import os
import json
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from langgraph.checkpoint.sqlite import SqliteSaver

# Base state directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATE_DIR = BASE_DIR / "state"
PROGRESS_JSON_PATH = STATE_DIR / "progress.json"
CHECKPOINTS_DB_PATH = STATE_DIR / "checkpoints.db"


def ensure_state_dir_exists() -> None:
    """Ensure that the state directory exists."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def get_sqlite_checkpointer(db_path: Optional[Path] = None) -> SqliteSaver:
    """
    Initialize and return a durable SQLite checkpointer for LangGraph.
    """
    ensure_state_dir_exists()
    target_path = db_path or CHECKPOINTS_DB_PATH
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    return SqliteSaver(conn)


def atomic_write_json(file_path: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON data to a file atomically via a temporary file.
    Ensures state integrity even if interrupted by SIGINT / Ctrl+C.
    """
    ensure_state_dir_exists()
    temp_path = file_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    temp_path.replace(file_path)


def read_progress_json(file_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Read and parse the human-readable progress.json file."""
    target_path = file_path or PROGRESS_JSON_PATH
    if not target_path.exists():
        return None
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to read {target_path}: {e}")
        return None


def save_progress_json(state: Dict[str, Any], file_path: Optional[Path] = None) -> None:
    """
    Update state/progress.json with the current workflow state.
    """
    target_path = file_path or PROGRESS_JSON_PATH
    
    # Calculate last completed item and total completed
    completed_items = [
        item_id for item_id, data in state.get("items", {}).items()
        if data.get("status") == "completed"
    ]
    last_completed = completed_items[-1] if completed_items else None
    
    progress_data = {
        "run_id": state.get("run_id", "assignment2-demo"),
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_llm_calls": state.get("total_llm_calls", 0),
        "last_completed_item": last_completed,
        "validation_complete": state.get("validation_complete", False),
        "items": state.get("items", {}),
        "validation_results": state.get("validation_results"),
    }
    
    atomic_write_json(target_path, progress_data)


def corrupt_item_result(
    item_id: str = "item3",
    bad_result: str = "This item is about cooking recipes and has nothing to do with the requested technical concept.",
    file_path: Optional[Path] = None,
) -> bool:
    """
    Intentionally corrupt an item's stored result in progress.json for validation testing.
    """
    data = read_progress_json(file_path)
    if not data or "items" not in data or item_id not in data["items"]:
        print(f"[Error] Cannot corrupt {item_id}: Item not found in persisted state.")
        return False
    
    data["items"][item_id]["result"] = bad_result
    data["items"][item_id]["status"] = "completed"
    atomic_write_json(file_path or PROGRESS_JSON_PATH, data)
    return True


def reset_all_state(state_dir: Optional[Path] = None) -> None:
    """
    Safely remove existing checkpoints database and progress.json.
    """
    target_dir = state_dir or STATE_DIR
    if not target_dir.exists():
        return
        
    for fname in ["progress.json", "progress.tmp", "checkpoints.db", "checkpoints.db-journal", "checkpoints.db-wal", "checkpoints.db-shm"]:
        p = target_dir / fname
        if p.exists():
            try:
                p.unlink()
            except Exception as e:
                print(f"[Warning] Could not delete {p}: {e}")
