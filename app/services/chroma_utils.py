from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


def clear_chroma_system_cache() -> None:
    """Clear Chroma's in-process shared system cache after a stale Rust binding."""
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def is_stale_chroma_error(exc: Exception) -> bool:
    """Detect the Chroma Rust binding state that appears after Streamlit reruns."""
    message = str(exc).lower()
    return "rustbindingsapi" in message and "bindings" in message


def create_persistent_client(persist_dir: str | Path, *, reset_cache: bool = False) -> Any:
    """Create a Chroma PersistentClient, retrying once if Chroma cached a stale system."""
    if reset_cache:
        clear_chroma_system_cache()

    try:
        return chromadb.PersistentClient(path=str(Path(persist_dir)))
    except Exception as exc:
        if is_stale_chroma_error(exc):
            clear_chroma_system_cache()
            return chromadb.PersistentClient(path=str(Path(persist_dir)))
        raise


def get_or_create_collection(
    persist_dir: str | Path,
    collection_name: str,
    *,
    metadata: dict[str, Any] | None = None,
    reset_cache: bool = False,
) -> tuple[Any, Any]:
    """Create a Chroma client and collection, retrying if the cached Rust binding is stale."""
    client = create_persistent_client(persist_dir, reset_cache=reset_cache)

    try:
        collection = client.get_or_create_collection(name=collection_name, metadata=metadata)
        collection.count()
        return client, collection
    except Exception as exc:
        if is_stale_chroma_error(exc):
            close_chroma_client(client)
            clear_chroma_system_cache()
            client = create_persistent_client(persist_dir)
            collection = client.get_or_create_collection(name=collection_name, metadata=metadata)
            collection.count()
            return client, collection
        raise


def close_chroma_client(client: Any) -> None:
    """Release Chroma resources when the installed version exposes close()."""
    close = getattr(client, "close", None)
    if callable(close):
        close()
