from __future__ import annotations

import asyncio
from functools import lru_cache

import chromadb
from groq import AsyncGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import Settings, get_settings


TOP_K = 3


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    settings = get_settings()
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    return Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
    )


@lru_cache(maxsize=1)
def get_groq_client() -> AsyncGroq:
    settings = get_settings()
    return AsyncGroq(api_key=settings.groq_api_key)


def _retrieve_context_documents(user_query: str, top_k: int) -> list:
    vector_store = get_vector_store()
    return vector_store.similarity_search(user_query, k=top_k)


def _format_context(documents: list) -> str:
    if not documents:
        return "Khong tim thay thong tin lien quan trong ChromaDB."

    formatted_chunks: list[str] = []
    for index, document in enumerate(documents, start=1):
        page = document.metadata.get("page", "unknown")
        source = document.metadata.get("source", "unknown")
        formatted_chunks.append(
            f"[Doan {index} | Trang {page} | Nguon: {source}]\n{document.page_content}"
        )

    return "\n\n".join(formatted_chunks)


def _build_system_prompt(context: str) -> str:
    return (
        "Ban la tro ly ao DLU. Hay dung thong tin sau day de tra loi sinh vien: "
        f"{context}. Neu thong tin khong co, hay tra loi la em khong biet."
    )


async def get_ai_response(user_query: str) -> str:
    """Retrieve top-k context from ChromaDB and generate a final answer with Groq."""
    cleaned_query = user_query.strip()
    if not cleaned_query:
        return "Em khong biet vi ban chua gui cau hoi hop le."

    try:
        documents = await asyncio.to_thread(_retrieve_context_documents, cleaned_query, TOP_K)
    except Exception:
        documents = []

    context = _format_context(documents)

    settings: Settings = get_settings()
    groq_client = get_groq_client()
    completion = await groq_client.chat.completions.create(
        model=settings.groq_model,
        temperature=0.2,
        max_completion_tokens=512,
        messages=[
            {"role": "system", "content": _build_system_prompt(context)},
            {"role": "user", "content": cleaned_query},
        ],
    )

    answer = completion.choices[0].message.content
    return answer.strip() if answer else "Em khong biet."
