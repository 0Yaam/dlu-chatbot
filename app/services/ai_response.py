from __future__ import annotations

import asyncio
import re
import unicodedata
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings
from app.services.chroma_utils import clear_chroma_system_cache, create_persistent_client, is_stale_chroma_error


TOP_K = 3
MIN_RELEVANCE_SCORE = 0.2
FALLBACK_ANSWER = "không tìm thấy câu trả lời trong dữ liệu"
TOKEN_STOPWORDS = {
    "la",
    "là",
    "va",
    "và",
    "cua",
    "của",
    "cho",
    "trong",
    "ngoai",
    "ngoài",
    "tai",
    "tại",
    "o",
    "ở",
    "tu",
    "từ",
    "den",
    "đến",
    "hoc",
    "học",
    "sinh",
    "vien",
    "viên",
    "gi",
    "gì",
    "nao",
    "nào",
    "nhung",
    "những",
    "the",
    "thể",
    "co",
    "có",
    "khong",
    "không",
    "duoc",
    "được",
    "xem",
    "ve",
    "về",
}


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )


@lru_cache(maxsize=8)
def get_vector_store(collection_name: str | None = None) -> Chroma:
    settings = get_settings()
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    client = create_persistent_client(settings.chroma_persist_dir)
    return Chroma(
        client=client,
        collection_name=collection_name or settings.chroma_collection,
        embedding_function=get_embeddings(),
    )


@lru_cache(maxsize=1)
def get_openrouter_client() -> AsyncOpenAI:
    settings = get_settings()
    headers: dict[str, str] = {}
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_site_name:
        headers["X-OpenRouter-Title"] = settings.openrouter_site_name
    return AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        default_headers=headers or None,
    )


def _retrieve_context_documents(user_query: str, top_k: int, collection_name: str | None = None) -> list:
    vector_store = get_vector_store(collection_name)
    try:
        return vector_store.similarity_search(user_query, k=top_k)
    except Exception as exc:
        if not is_stale_chroma_error(exc):
            raise
        get_vector_store.cache_clear()
        clear_chroma_system_cache()
        vector_store = get_vector_store(collection_name)
        return vector_store.similarity_search(user_query, k=top_k)


def _retrieve_context_documents_with_scores(
    user_query: str,
    top_k: int,
    collection_name: str | None = None,
) -> list[tuple[Any, float]]:
    vector_store = get_vector_store(collection_name)
    try:
        return vector_store.similarity_search_with_relevance_scores(user_query, k=top_k)
    except Exception as exc:
        if not is_stale_chroma_error(exc):
            raise
        get_vector_store.cache_clear()
        clear_chroma_system_cache()
        vector_store = get_vector_store(collection_name)
        return vector_store.similarity_search_with_relevance_scores(user_query, k=top_k)


def _format_context(documents: list) -> str:
    if not documents:
        return "Không tìm thấy thông tin liên quan trong ChromaDB."

    formatted_chunks: list[str] = []
    for index, document in enumerate(documents, start=1):
        page = document.metadata.get("page", "unknown")
        source = document.metadata.get("source", "unknown")
        formatted_chunks.append(
            f"[Đoạn {index} | Trang {page} | Nguồn: {source}]\n{document.page_content}"
        )

    return "\n\n".join(formatted_chunks)


def _build_system_prompt(context: str) -> str:
    return (
        "Bạn là trợ lý ảo DLU. "
        "Chỉ được phép trả lời dựa trên context đã cung cấp. "
        "Nếu context không chứa câu trả lời, không đủ thông tin, hoặc không liên quan, "
        f'hãy trả lời đúng nguyên văn: "{FALLBACK_ANSWER}". '
        "Không được bổ sung kiến thức bên ngoài context. "
        f"Context: {context}"
    )


async def _generate_answer_from_context(user_query: str, context: str) -> str:
    """Send the final prompt to OpenRouter and return the generated answer."""
    settings: Settings = get_settings()
    openrouter_client = get_openrouter_client()
    completion = await openrouter_client.chat.completions.create(
        model=settings.openrouter_model,
        temperature=0.2,
        max_tokens=512,
        messages=[
            {"role": "system", "content": _build_system_prompt(context)},
            {"role": "user", "content": user_query},
        ],
    )

    answer = completion.choices[0].message.content
    return answer.strip() if answer else FALLBACK_ANSWER


def _normalize_similarity_score(score: float) -> float:
    """Clamp similarity scores to a friendly 0-1 range for the inspector UI."""
    return max(0.0, min(1.0, float(score)))


def _strip_accents(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def _tokenize(text: str) -> set[str]:
    normalized = _strip_accents(text.lower())
    tokens = set(re.findall(r"\w+", normalized))
    return {
        token
        for token in tokens
        if len(token) >= 2 and token not in TOKEN_STOPWORDS and not token.isdigit()
    }


def _has_query_overlap(user_query: str, retrieved_items: list[tuple[Any, float]]) -> bool:
    """Check whether query keywords actually appear in the retrieved content."""
    query_tokens = _tokenize(user_query)
    if not query_tokens:
        return False

    for document, _score in retrieved_items:
        content_tokens = _tokenize(document.page_content)
        metadata_tokens = _tokenize(" ".join(str(value) for value in document.metadata.values()))
        if query_tokens & (content_tokens | metadata_tokens):
            return True
    return False


def _filter_relevant_items(retrieved_items: list[tuple[Any, float]]) -> list[tuple[Any, float]]:
    """Keep only chunks that are sufficiently relevant to answer from data."""
    return [
        (document, score)
        for document, score in retrieved_items
        if _normalize_similarity_score(score) >= MIN_RELEVANCE_SCORE
    ]


def _serialize_chunk(rank: int, document: Any, score: float) -> dict[str, Any]:
    """Convert a retrieved document into a dashboard-friendly structure."""
    normalized_score = _normalize_similarity_score(score)
    metadata = dict(document.metadata)
    return {
        "rank": rank,
        "score": normalized_score,
        "score_percent": round(normalized_score * 100, 2),
        "source": metadata.get("source", "unknown"),
        "page": metadata.get("page", "unknown"),
        "section_title": metadata.get("section_title", ""),
        "chunk_method": metadata.get("chunk_method", ""),
        "metadata": metadata,
        "content": document.page_content,
    }


async def inspect_rag_pipeline(
    user_query: str,
    *,
    top_k: int = TOP_K,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """Expose the retrieval + generation flow for the Streamlit inspector."""
    settings = get_settings()
    cleaned_query = user_query.strip()
    selected_collection = collection_name or settings.chroma_collection

    if not cleaned_query:
        return {
            "query": "",
            "chunks": [],
            "answer": FALLBACK_ANSWER,
            "context": "Không tìm thấy thông tin liên quan trong ChromaDB.",
            "top_k": top_k,
            "collection_name": selected_collection,
            "model": settings.openrouter_model,
            "fallback_used": True,
            "context_sufficient": False,
            "relevance_threshold": MIN_RELEVANCE_SCORE,
        }

    retrieved_items = await asyncio.to_thread(
        _retrieve_context_documents_with_scores,
        cleaned_query,
        top_k,
        selected_collection,
    )
    relevant_items = _filter_relevant_items(retrieved_items)
    has_overlap = _has_query_overlap(cleaned_query, relevant_items or retrieved_items)
    documents = [document for document, _score in relevant_items]
    context = _format_context(documents)

    if not relevant_items or not has_overlap:
        answer = FALLBACK_ANSWER
    else:
        answer = await _generate_answer_from_context(cleaned_query, context)
        normalized_answer = answer.strip().lower().rstrip(".! ")
        if FALLBACK_ANSWER in normalized_answer:
            answer = FALLBACK_ANSWER

    fallback_used = not bool(relevant_items) or not has_overlap or answer == FALLBACK_ANSWER

    return {
        "query": cleaned_query,
        "chunks": [
            _serialize_chunk(rank=index, document=document, score=score)
            for index, (document, score) in enumerate(retrieved_items, start=1)
        ],
        "answer": answer,
        "context": context,
        "top_k": top_k,
        "collection_name": selected_collection,
        "model": settings.openrouter_model,
        "fallback_used": fallback_used,
        "context_sufficient": bool(relevant_items) and has_overlap and not fallback_used,
        "relevance_threshold": MIN_RELEVANCE_SCORE,
    }


async def get_ai_response(user_query: str) -> str:
    """Retrieve top-k context from ChromaDB and generate a final answer with OpenRouter."""
    result = await inspect_rag_pipeline(user_query, top_k=TOP_K)
    return result["answer"]
