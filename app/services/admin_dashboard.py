from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from openai import OpenAI


logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
ENV_PATH = Path(".env")
HISTORY_PATH = DATA_DIR / "chat_history.jsonl"
ENV_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


def append_conversation_history(
    question: str,
    answer: str,
    response_time_ms: float,
    *,
    status: str = "success",
    user_id: int | None = None,
    username: str | None = None,
    full_name: str | None = None,
    chat_id: int | None = None,
    history_path: str | Path = HISTORY_PATH,
) -> None:
    """Persist a lightweight chat log for the admin dashboard."""
    target_path = Path(history_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "question": question,
        "answer": answer,
        "response_time_ms": round(response_time_ms, 2),
        "user_id": user_id,
        "username": username or "",
        "full_name": full_name or "",
        "chat_id": chat_id,
    }

    with target_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_conversation_history(
    history_path: str | Path = HISTORY_PATH,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load dashboard chat history from a JSONL file."""
    target_path = Path(history_path)
    if not target_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with target_path.open("r", encoding="utf-8") as file:
        for line in file:
            cleaned_line = line.strip()
            if not cleaned_line:
                continue
            try:
                records.append(json.loads(cleaned_line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed chat history line: %s", cleaned_line[:120])

    records.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    if limit is not None:
        return records[:limit]
    return records


def get_pdf_inventory(data_dir: str | Path = DATA_DIR) -> list[dict[str, Any]]:
    """Return basic metadata for PDF files in the knowledge directory."""
    source_dir = Path(data_dir)
    if not source_dir.exists():
        return []

    pdf_rows: list[dict[str, Any]] = []
    for pdf_path in sorted(source_dir.glob("*.pdf")):
        stats = pdf_path.stat()
        pdf_rows.append(
            {
                "name": pdf_path.name,
                "path": str(pdf_path),
                "size_bytes": stats.st_size,
                "modified_at": datetime.fromtimestamp(stats.st_mtime),
            }
        )

    return pdf_rows


def format_bytes(size_bytes: int) -> str:
    """Human-readable file size formatting."""
    if size_bytes <= 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def compute_dashboard_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate conversation metrics for the overview cards."""
    total_questions = len(records)
    response_values = [
        float(record.get("response_time_ms", 0))
        for record in records
        if isinstance(record.get("response_time_ms"), (int, float))
    ]
    successful_questions = sum(1 for record in records if record.get("status") == "success")
    failed_questions = sum(1 for record in records if record.get("status") != "success")
    unique_users = len({record.get("user_id") for record in records if record.get("user_id") is not None})

    average_response_ms = round(sum(response_values) / len(response_values), 2) if response_values else 0.0
    return {
        "total_questions": total_questions,
        "average_response_ms": average_response_ms,
        "successful_questions": successful_questions,
        "failed_questions": failed_questions,
        "unique_users": unique_users,
    }


def load_env_values(env_path: str | Path = ENV_PATH) -> dict[str, str]:
    """Parse a simple .env file without requiring extra dependencies."""
    target_path = Path(env_path)
    if not target_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in target_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        match = ENV_PATTERN.match(raw_line)
        if match:
            key, value = match.groups()
            values[key] = value.strip().strip("'\"")

    return values


def update_env_values(updates: dict[str, str], env_path: str | Path = ENV_PATH) -> None:
    """Update or append environment variables while preserving comments."""
    target_path = Path(env_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines = target_path.read_text(encoding="utf-8").splitlines() if target_path.exists() else []

    new_lines: list[str] = []
    updated_keys: set[str] = set()

    for line in existing_lines:
        match = ENV_PATTERN.match(line)
        if not match:
            new_lines.append(line)
            continue

        key = match.group(1)
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            updated_keys.add(key)
        else:
            new_lines.append(line)

    for key, value in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}")

    target_path.write_text("\n".join(new_lines).strip() + "\n", encoding="utf-8")


def mask_secret(secret: str, visible_chars: int = 4) -> str:
    """Mask secrets for safe dashboard display."""
    if not secret:
        return "Chua cau hinh"
    if len(secret) <= visible_chars * 2:
        return "*" * len(secret)
    return f"{secret[:visible_chars]}{'*' * max(4, len(secret) - visible_chars * 2)}{secret[-visible_chars:]}"


def check_openrouter_connection(
    api_key: str | None,
    model_name: str | None = None,
    *,
    site_url: str = "",
    site_name: str = "DLU Chatbot",
) -> dict[str, Any]:
    """Validate OpenRouter API reachability with the same SDK family used by runtime."""
    if not api_key:
        return {"ok": False, "label": "Chua cau hinh", "detail": "Thieu OPENROUTER_API_KEY"}

    try:
        headers: dict[str, str] = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-OpenRouter-Title"] = site_name
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers=headers or None,
        )
        selected_model = model_name or "meta-llama/llama-3.1-8b-instruct"
        client.chat.completions.create(
            model=selected_model,
            max_tokens=1,
            temperature=0,
            messages=[
                {"role": "system", "content": "Reply with OK only."},
                {"role": "user", "content": "health"},
            ],
        )
        detail = f"Da ket noi. Model {selected_model} san sang."
        return {"ok": True, "label": "Da ket noi", "detail": detail}
    except Exception as exc:
        message = str(exc)
        if "401" in message:
            detail = "Xac thuc that bai"
        elif "403" in message:
            detail = "Bi tu choi truy cap"
        else:
            detail = message
        return {"ok": False, "label": "Loi ket noi", "detail": detail}


def check_chromadb_connection(collection_name: str, persist_dir: str | Path) -> dict[str, Any]:
    """Check local ChromaDB health and collection size."""
    try:
        client = chromadb.PersistentClient(path=str(Path(persist_dir)))
        collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        vector_count = collection.count()
        return {
            "ok": True,
            "label": "San sang",
            "detail": f"{vector_count} vectors trong collection {collection_name}",
            "vector_count": vector_count,
        }
    except Exception as exc:
        return {"ok": False, "label": "Khong san sang", "detail": str(exc), "vector_count": 0}


def list_chroma_collections(persist_dir: str | Path) -> list[dict[str, Any]]:
    """List local ChromaDB collections with document counts for admin views."""
    try:
        client = chromadb.PersistentClient(path=str(Path(persist_dir)))
        collections = []
        for collection in client.list_collections():
            collections.append(
                {
                    "name": collection.name,
                    "vector_count": client.get_collection(collection.name).count(),
                }
            )
        collections.sort(key=lambda item: item["name"])
        return collections
    except Exception as exc:
        logger.warning("Unable to list ChromaDB collections: %s", exc)
        return []
