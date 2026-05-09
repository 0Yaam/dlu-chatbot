from __future__ import annotations

import asyncio
import sys
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.admin_dashboard import (
    check_chromadb_connection,
    check_openrouter_connection,
    compute_dashboard_metrics,
    format_bytes,
    get_pdf_inventory,
    list_chroma_collections,
    load_conversation_history,
    load_env_values,
    mask_secret,
    update_env_values,
)
from app.services.ai_response import (
    get_embeddings,
    get_openrouter_client,
    get_vector_store,
    inspect_rag_pipeline,
)
from app.services.chroma_utils import clear_chroma_system_cache
from app.services.pdf_indexer import DLUKnowledgeIndexer


st.set_page_config(
    page_title="DLU Dashboard",
    page_icon=":material/dashboard:",
    layout="wide",
    initial_sidebar_state="expanded",
)


PAGES: list[tuple[str, str, str]] = [
    ("overview", "Tổng quan hệ thống", "fa-solid fa-chart-line"),
    ("inspector", "RAG Inspector", "fa-solid fa-magnifying-glass-chart"),
    ("knowledge", "Quản lý tri thức", "fa-solid fa-folder-open"),
    ("history", "Lịch sử hội thoại", "fa-solid fa-comments"),
    ("config", "Cấu hình AI", "fa-solid fa-sliders"),
]


def inject_styles() -> None:
    """Load a modern minimalist design system for the Streamlit dashboard."""
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
        <style>
            :root {
                --blue-900: #1E3A8A;
                --blue-700: #1D4ED8;
                --blue-100: #DBEAFE;
                --gray-25: #FCFCFD;
                --gray-50: #F9FAFB;
                --gray-100: #F3F4F6;
                --gray-200: #E5E7EB;
                --gray-300: #D1D5DB;
                --gray-500: #6B7280;
                --gray-700: #374151;
                --gray-900: #111827;
                --shadow-soft: 0 10px 30px rgba(15, 23, 42, 0.05);
                --shadow-hover: 0 14px 34px rgba(15, 23, 42, 0.08);
                --radius-md: 12px;
                --radius-lg: 18px;
            }

            html, body, [class*="css"], [data-testid="stMarkdownContainer"], p, label, small {
                font-family: "Segoe UI", Arial, sans-serif !important;
                color: var(--gray-900);
            }

            .stApp {
                background: linear-gradient(180deg, #F7F8FA 0%, #FFFFFF 22%, #F7F8FA 100%);
                color: var(--gray-900);
            }

            .block-container {
                max-width: 1320px;
                padding-top: 1.4rem;
                padding-bottom: 2rem;
            }

            section[data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--gray-200);
            }

            section[data-testid="stSidebar"] .block-container {
                padding-top: 1.2rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero-panel {
                background: #ffffff;
                border: 1px solid var(--gray-200);
                border-left: 4px solid var(--blue-900);
                border-radius: 18px;
                box-shadow: var(--shadow-soft);
                padding: 1.25rem 1.35rem;
                color: var(--gray-900);
            }

            .hero-panel::after {
                display: none;
            }

            .hero-kicker {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.34rem 0.62rem;
                border-radius: 999px;
                background: var(--gray-100);
                color: var(--blue-900);
                font-size: 0.78rem;
                font-weight: 600;
                letter-spacing: 0.01em;
                margin-bottom: 0.75rem;
            }

            .hero-title {
                margin: 0;
                font-size: 1.75rem;
                line-height: 1.2;
                font-weight: 700;
                letter-spacing: -0.02em;
                color: var(--gray-900);
            }

            .hero-subtitle {
                margin: 0.55rem 0 0 0;
                max-width: 860px;
                font-size: 0.95rem;
                line-height: 1.7;
                color: var(--gray-900);
            }

            .surface-card,
            .stat-card,
            .chunk-card,
            .notice-card,
            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: var(--radius-md) !important;
                transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
            }

            .surface-card,
            .stat-card,
            .notice-card,
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid var(--gray-200);
                box-shadow: var(--shadow-soft);
                height: 100%;
            }

            .surface-card:hover,
            .stat-card:hover,
            .chunk-card:hover,
            div[data-testid="stVerticalBlockBorderWrapper"]:hover {
                transform: none;
                box-shadow: var(--shadow-soft);
            }

            .stat-card {
                padding: 1rem 1.05rem;
                min-height: 148px;
            }

            .stat-topline {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.95rem;
            }

            .stat-label {
                color: var(--gray-900);
                font-size: 0.9rem;
                margin: 0;
                font-weight: 600;
            }

            .stat-icon {
                width: 40px;
                height: 40px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: var(--blue-100);
                color: var(--blue-900);
                font-size: 0.98rem;
            }

            .stat-value {
                margin: 0;
                font-size: 1.85rem;
                font-weight: 700;
                letter-spacing: -0.03em;
                color: var(--gray-900);
            }

            .stat-caption {
                margin-top: 0.42rem;
                color: var(--gray-900);
                font-size: 0.86rem;
                line-height: 1.55;
            }

            .surface-card {
                padding: 1.12rem 1.18rem;
            }

            .section-title {
                margin: 0 0 0.2rem 0;
                font-size: 1.02rem;
                font-weight: 700;
                color: var(--gray-900);
            }

            .section-caption {
                margin: 0 0 0.95rem 0;
                color: var(--gray-900);
                font-size: 0.9rem;
                line-height: 1.6;
            }

            .chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
            }

            .chip {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.38rem 0.72rem;
                border-radius: 999px;
                border: 1px solid var(--gray-200);
                background: var(--gray-50);
                color: var(--gray-900);
                font-size: 0.82rem;
                font-weight: 500;
            }

            .notice-card {
                padding: 0.92rem 1rem;
                display: flex;
                align-items: flex-start;
                gap: 0.78rem;
                margin: 0.4rem 0 1rem 0;
            }

            .notice-card.notice-info {
                border-left: 4px solid var(--blue-900);
            }

            .notice-card.notice-success {
                border-left: 4px solid #0F766E;
            }

            .notice-card.notice-warning {
                border-left: 4px solid #B45309;
            }

            .notice-card.notice-error {
                border-left: 4px solid #B91C1C;
            }

            .notice-icon {
                width: 34px;
                height: 34px;
                flex-shrink: 0;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: var(--gray-100);
                color: var(--blue-900);
                margin-top: 0.05rem;
            }

            .notice-title {
                margin: 0 0 0.15rem 0;
                font-size: 0.94rem;
                font-weight: 700;
                color: var(--gray-900);
            }

            .notice-text {
                margin: 0;
                font-size: 0.88rem;
                line-height: 1.6;
                color: var(--gray-900);
            }

            .sidebar-brand {
                padding: 0.95rem 1rem;
                border-radius: 16px;
                background: #ffffff;
                border: 1px solid var(--gray-200);
                border-left: 4px solid var(--blue-900);
                color: var(--gray-900);
                box-shadow: var(--shadow-soft);
                margin-bottom: 0.95rem;
            }

            .sidebar-brand-title {
                margin: 0 0 0.2rem 0;
                font-size: 1.12rem;
                font-weight: 700;
            }

            .sidebar-brand-text {
                margin: 0;
                font-size: 0.84rem;
                line-height: 1.55;
                color: var(--gray-900);
            }

            .sidebar-icon-wrap {
                width: 38px;
                height: 38px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: var(--gray-100);
                color: var(--blue-900);
                border: 1px solid var(--gray-200);
            }

            .sidebar-caption {
                color: var(--gray-900);
                font-size: 0.82rem;
                line-height: 1.55;
                margin-top: 0.85rem;
            }

            .stButton > button,
            .stDownloadButton > button,
            .stFormSubmitButton > button {
                font-family: "Segoe UI", Arial, sans-serif !important;
                border-radius: 12px !important;
                border: 1px solid var(--gray-300) !important;
                background: #ffffff !important;
                color: var(--gray-900) !important;
                font-weight: 600 !important;
                padding: 0.58rem 0.9rem !important;
                min-height: 2.8rem !important;
                box-shadow: none !important;
                transition: none !important;
            }

            .stButton > button:hover,
            .stDownloadButton > button:hover,
            .stFormSubmitButton > button:hover {
                border-color: var(--gray-300) !important;
                background: #ffffff !important;
                color: var(--gray-900) !important;
                transform: none !important;
            }

            .stButton > button[kind="primary"],
            .stFormSubmitButton > button[kind="primary"] {
                background: #ffffff !important;
                border-color: var(--blue-900) !important;
                color: var(--gray-900) !important;
            }

            .stButton > button[kind="primary"]:hover,
            .stFormSubmitButton > button[kind="primary"]:hover {
                background: #ffffff !important;
                border-color: var(--blue-900) !important;
                color: var(--gray-900) !important;
            }

            .stTextInput > div > div,
            .stTextArea textarea,
            .stSelectbox [data-baseweb="select"] > div,
            .stMultiSelect [data-baseweb="select"] > div {
                border-radius: 12px !important;
                border-color: var(--gray-300) !important;
                background: #ffffff !important;
            }

            .stTextInput > div > div:focus-within,
            .stTextArea textarea:focus,
            .stSelectbox [data-baseweb="select"] > div:focus-within,
            .stMultiSelect [data-baseweb="select"] > div:focus-within {
                border-color: var(--blue-900) !important;
                box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.08) !important;
            }

            .stTextInput input,
            .stTextArea textarea,
            .stTextInput input::placeholder,
            .stTextArea textarea::placeholder,
            .stSelectbox [data-baseweb="select"] *:not([data-testid="stIconMaterial"]),
            .stMultiSelect [data-baseweb="select"] *:not([data-testid="stIconMaterial"]) {
                font-family: "Segoe UI", Arial, sans-serif !important;
                color: var(--gray-900) !important;
                -webkit-text-fill-color: var(--gray-900) !important;
                opacity: 1 !important;
            }

            .stSelectbox svg,
            .stMultiSelect svg {
                fill: var(--gray-900) !important;
            }

            .stSlider [data-baseweb="slider"] > div div[role="slider"] {
                background: var(--blue-900) !important;
                border-color: var(--blue-900) !important;
            }

            .stSlider [data-baseweb="slider"] > div div:nth-child(1) {
                background: var(--blue-100) !important;
            }

            div[data-testid="stFileUploader"] {
                border-radius: 14px;
            }

            div[data-testid="stFileUploader"] section {
                border-radius: 14px !important;
                border: 1.5px dashed var(--gray-900) !important;
                background: #ffffff !important;
                padding: 0.85rem !important;
                transition: none !important;
            }

            div[data-testid="stFileUploader"] section:hover {
                border-color: var(--gray-900) !important;
                background: #ffffff !important;
            }

            div[data-testid="stFileUploader"] button {
                font-family: "Segoe UI", Arial, sans-serif !important;
                background: #ffffff !important;
                border: 1px solid var(--gray-300) !important;
                border-color: var(--gray-300) !important;
                color: var(--gray-900) !important;
                box-shadow: none !important;
            }

            div[data-testid="stFileUploader"] button:hover {
                background: #ffffff !important;
                border-color: var(--gray-300) !important;
                color: var(--gray-900) !important;
            }

            div[data-testid="stFileUploader"] button *:not([data-testid="stIconMaterial"]),
            div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] *:not([data-testid="stIconMaterial"]),
            div[data-testid="stFileUploader"] label,
            div[data-testid="stFileUploader"] small,
            div[data-testid="stFileUploader"] span:not([data-testid="stIconMaterial"]) {
                font-family: "Segoe UI", Arial, sans-serif !important;
                color: var(--gray-900) !important;
                -webkit-text-fill-color: var(--gray-900) !important;
                opacity: 1 !important;
            }

            div[data-testid="stFileUploaderDropzoneInstructions"] small,
            div[data-testid="stFileUploaderDropzoneInstructions"] span,
            div[data-testid="stFileUploaderDropzoneInstructions"] div {
                color: var(--gray-900) !important;
            }

            div[data-testid="stFileUploader"] svg,
            div[data-testid="stFileUploader"] path {
                fill: var(--gray-900) !important;
                stroke: var(--gray-900) !important;
            }

            section[data-testid="stSidebar"] .stButton > button,
            section[data-testid="stSidebar"] .stButton > button[kind="primary"],
            section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
                background: #ffffff !important;
                color: var(--gray-900) !important;
                border: 1px solid var(--gray-300) !important;
                box-shadow: none !important;
            }

            section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
                border-color: var(--blue-900) !important;
                font-weight: 700 !important;
            }

            section[data-testid="stSidebar"] .stButton > button:hover,
            section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
            section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
                background: #ffffff !important;
                color: var(--gray-900) !important;
                border-color: var(--gray-300) !important;
                transform: none !important;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid var(--gray-200);
            }

            div[data-testid="stExpander"] {
                background: #ffffff;
                border: 1px solid var(--gray-200);
                border-radius: 12px !important;
                overflow: hidden;
                margin-bottom: 0.7rem;
            }

            div[data-testid="stExpander"] details summary {
                background: #ffffff;
            }

            div[data-testid="stExpander"] details summary p {
                font-family: "Segoe UI", Arial, sans-serif !important;
                font-size: 0.92rem;
                font-weight: 600;
                color: var(--gray-900);
            }

            div[data-testid="stExpander"] details summary,
            div[data-testid="stExpander"] details summary *:not([data-testid="stIconMaterial"]) {
                color: var(--gray-900) !important;
                -webkit-text-fill-color: var(--gray-900) !important;
                font-family: "Segoe UI", Arial, sans-serif !important;
            }

            [data-testid="stIconMaterial"] {
                font-family: "Material Symbols Rounded" !important;
                font-weight: normal !important;
                font-style: normal !important;
                font-size: 1.1rem !important;
                line-height: 1 !important;
                letter-spacing: normal !important;
                text-transform: none !important;
                white-space: nowrap !important;
                word-wrap: normal !important;
                direction: ltr !important;
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                -webkit-font-smoothing: antialiased !important;
                font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24 !important;
            }

            .chunk-card {
                border-radius: 12px;
                border: 1px solid var(--gray-200);
                padding: 1rem;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.5);
            }

            .chunk-tone-0 { background: #F8FAFC; }
            .chunk-tone-1 { background: #EFF6FF; }
            .chunk-tone-2 { background: #F5F3FF; }
            .chunk-tone-3 { background: #EEF2FF; }

            .chunk-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.75rem;
                margin-bottom: 0.8rem;
                flex-wrap: wrap;
            }

            .chunk-title {
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 0.5rem;
            }

            .file-pill,
            .score-pill,
            .meta-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                border-radius: 999px;
                padding: 0.28rem 0.62rem;
                font-size: 0.78rem;
                font-weight: 600;
                border: 1px solid rgba(30, 58, 138, 0.12);
            }

            .file-pill {
                background: rgba(30, 58, 138, 0.08);
                color: var(--blue-900);
            }

            .score-pill,
            .meta-pill {
                background: rgba(255, 255, 255, 0.72);
                color: var(--gray-700);
            }

            .chunk-content {
                color: var(--gray-900);
                font-size: 0.92rem;
                line-height: 1.7;
                white-space: pre-wrap;
            }

            .empty-state {
                border-radius: 12px;
                padding: 1rem 1.05rem;
                background: var(--gray-50);
                border: 1px dashed var(--gray-300);
                color: var(--gray-900);
                font-size: 0.9rem;
                line-height: 1.6;
            }

            .summary-list {
                display: grid;
                gap: 0.75rem;
            }

            .summary-item {
                display: flex;
                gap: 0.8rem;
                align-items: flex-start;
                padding: 0.9rem 0.95rem;
                border: 1px solid var(--gray-200);
                border-radius: 12px;
                background: var(--gray-50);
            }

            .summary-item-icon {
                width: 36px;
                height: 36px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #E5ECFB;
                color: var(--blue-900);
                flex-shrink: 0;
            }

            .summary-item-title {
                margin: 0 0 0.18rem 0;
                font-size: 0.92rem;
                font-weight: 700;
                color: var(--gray-900);
            }

            .summary-item-text {
                margin: 0;
                font-size: 0.88rem;
                line-height: 1.6;
                color: var(--gray-900);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clear_runtime_caches() -> None:
    """Clear runtime caches after configuration updates."""
    get_settings.cache_clear()
    get_embeddings.cache_clear()
    get_vector_store.cache_clear()
    get_openrouter_client.cache_clear()
    clear_chroma_system_cache()


def run_async_task(coroutine):
    """Run an async task safely from Streamlit."""
    try:
        return asyncio.run(coroutine)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coroutine)
        finally:
            loop.close()


def load_settings_snapshot() -> dict[str, str]:
    """Return settings with a safe fallback when .env is incomplete."""
    env_values = load_env_values()
    try:
        settings = get_settings()
        return {
            "openrouter_model": settings.openrouter_model,
            "openrouter_site_url": settings.openrouter_site_url,
            "openrouter_site_name": settings.openrouter_site_name,
            "chroma_collection": settings.chroma_collection,
            "chroma_persist_dir": str(settings.chroma_persist_dir),
            "embedding_model": settings.embedding_model,
            "embedding_device": settings.embedding_device,
        }
    except Exception:
        return {
            "openrouter_model": env_values.get("OPENROUTER_MODEL", env_values.get("GROQ_MODEL", "meta-llama/llama-3.1-8b-instruct")),
            "openrouter_site_url": env_values.get("OPENROUTER_SITE_URL", ""),
            "openrouter_site_name": env_values.get("OPENROUTER_SITE_NAME", "DLU Chatbot"),
            "chroma_collection": env_values.get("CHROMA_COLLECTION", "dlu_knowledge"),
            "chroma_persist_dir": env_values.get("CHROMA_PERSIST_DIR", "vector_store"),
            "embedding_model": env_values.get(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            "embedding_device": env_values.get("EMBEDDING_DEVICE", "cpu"),
        }


def set_active_page(page_key: str) -> None:
    """Persist the selected page in session state."""
    st.session_state["dashboard_active_page"] = page_key


def current_page_key() -> str:
    """Return the current dashboard page key."""
    if "dashboard_active_page" not in st.session_state:
        st.session_state["dashboard_active_page"] = PAGES[0][0]
    return st.session_state["dashboard_active_page"]


def render_header(icon: str, eyebrow: str, title: str, subtitle: str) -> None:
    """Render the dashboard hero section."""
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-kicker"><i class="{escape(icon)}"></i>{escape(eyebrow)}</div>
            <h1 class="hero-title">{escape(title)}</h1>
            <p class="hero-subtitle">{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notice(kind: str, title: str, message: str, icon: str | None = None) -> None:
    """Render custom HTML notices instead of Streamlit status boxes."""
    icon_map = {
        "info": "fa-solid fa-circle-info",
        "success": "fa-solid fa-circle-check",
        "warning": "fa-solid fa-triangle-exclamation",
        "error": "fa-solid fa-circle-exclamation",
    }
    chosen_icon = icon or icon_map.get(kind, icon_map["info"])
    st.markdown(
        f"""
        <div class="notice-card notice-{escape(kind)}">
            <div class="notice-icon"><i class="{escape(chosen_icon)}"></i></div>
            <div>
                <p class="notice-title">{escape(title)}</p>
                <p class="notice-text">{escape(message)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(title: str, value: str, caption: str, icon: str) -> None:
    """Render a compact dashboard metric card."""
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-topline">
                <p class="stat-label">{escape(title)}</p>
                <div class="stat-icon"><i class="{escape(icon)}"></i></div>
            </div>
            <p class="stat-value">{escape(value)}</p>
            <div class="stat-caption">{escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_surface_card(title: str, caption: str = "") -> None:
    """Open a styled information card."""
    body = f'<p class="section-caption">{escape(caption)}</p>' if caption else ""
    st.markdown(
        f"""
        <div class="surface-card">
            <h3 class="section-title">{escape(title)}</h3>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(metrics: dict[str, Any], pdf_count: int, collection_count: int) -> str:
    """Render a custom sidebar with FontAwesome icons."""
    active_page = current_page_key()

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <p class="sidebar-brand-title">DLU Admin Dashboard</p>
                <p class="sidebar-brand-text">
                    Modern Minimalist control panel cho đồ án RAG Chatbot sinh viên Đại học Đà Lạt.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for page_key, label, icon in PAGES:
            icon_col, button_col = st.columns([0.18, 0.82], gap="small")
            with icon_col:
                st.markdown(f'<div class="sidebar-icon-wrap"><i class="{escape(icon)}"></i></div>', unsafe_allow_html=True)
            with button_col:
                if st.button(
                    label,
                    key=f"nav_{page_key}",
                    width="stretch",
                    type="primary" if active_page == page_key else "secondary",
                ):
                    if active_page != page_key:
                        set_active_page(page_key)
                        st.rerun()

        st.markdown(
            f"""
            <p class="sidebar-caption">
                {metrics["total_questions"]} câu hỏi · {pdf_count} PDF · {collection_count} collections
            </p>
            """,
            unsafe_allow_html=True,
        )

    return current_page_key()


def run_indexing_workflow(settings_snapshot: dict[str, str]) -> None:
    """Run the PDF indexer with live progress and compact logs."""
    progress_holder = st.empty()
    status_holder = st.empty()
    log_holder = st.empty()
    logs: list[str] = []

    progress_bar = progress_holder.progress(0.0, text="Đang khởi tạo indexing...")

    def append_log(message: str) -> None:
        logs.append(message)
        log_holder.code("\n".join(logs[-160:]), language="text")

    def update_progress(value: float, status: str) -> None:
        progress_bar.progress(value, text=status)
        status_holder.markdown(
            f'<div class="chip"><i class="fa-solid fa-gear"></i>{escape(status)}</div>',
            unsafe_allow_html=True,
        )

    try:
        indexer = DLUKnowledgeIndexer(
            data_dir="data",
            persist_dir=settings_snapshot["chroma_persist_dir"],
            collection_name=settings_snapshot["chroma_collection"],
            embedding_model_name=settings_snapshot["embedding_model"],
            device=settings_snapshot["embedding_device"],
            log_callback=append_log,
            progress_callback=update_progress,
        )
        summary = indexer.run()
        clear_runtime_caches()

        st.session_state["knowledge_logs"] = logs
        st.session_state["knowledge_summary"] = summary
        progress_bar.progress(1.0, text="Indexing hoàn tất.")
        render_notice(
            "success",
            "Indexing hoàn tất",
            f"Collection {summary['collection_name']} hiện có {summary['vector_count']} vectors sau khi xử lý {summary['chunks_indexed']} chunks.",
        )
    except Exception as exc:
        st.session_state["knowledge_logs"] = logs
        st.session_state["knowledge_summary"] = None
        progress_bar.progress(1.0, text="Indexing dừng do lỗi.")
        append_log(f"Loi nghiem trong khi indexing: {exc}")
        render_notice("error", "Indexing thất bại", str(exc))


def render_overview_page(
    metrics: dict[str, Any],
    pdf_inventory: list[dict[str, Any]],
    openrouter_status: dict[str, Any],
    chroma_status: dict[str, Any],
    settings_snapshot: dict[str, str],
) -> None:
    """Render the overview dashboard."""
    render_header(
        "fa-solid fa-layer-group",
        "System Overview",
        "Tổng quan hệ thống",
        "Theo dõi nhanh tình trạng vận hành của chatbot RAG, số lượng tài liệu tri thức, độ trễ phản hồi và kết nối tới OpenRouter cùng ChromaDB.",
    )

    top_cards = st.columns(4)
    with top_cards[0]:
        render_stat_card("Số lượng PDF", str(len(pdf_inventory)), "Tệp tri thức đang có trong thư mục data.", "fa-regular fa-file-pdf")
    with top_cards[1]:
        render_stat_card("Tổng câu hỏi", str(metrics["total_questions"]), "Ghi nhận từ lịch sử hội thoại của bot.", "fa-solid fa-comments")
    with top_cards[2]:
        render_stat_card("Latency trung bình", f'{metrics["average_response_ms"]:.0f} ms', "Độ trễ phản hồi trung bình của chatbot.", "fa-solid fa-gauge-high")
    with top_cards[3]:
        online_label = "Online" if openrouter_status["ok"] and chroma_status["ok"] else "Cần kiểm tra"
        render_stat_card("Trạng thái hệ thống", online_label, "Tổng hợp OpenRouter API và ChromaDB.", "fa-solid fa-signal")

    st.write("")
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        st.markdown(
            """
            <div class="surface-card">
                <h3 class="section-title">Tóm tắt hệ thống</h3>
                <p class="section-caption">Chỉ giữ lại các khối cần thiết để nhìn vào là hiểu luồng vận hành.</p>
                <div class="summary-list">
                    <div class="summary-item">
                        <div class="summary-item-icon"><i class="fa-solid fa-paper-plane"></i></div>
                        <div>
                            <p class="summary-item-title">FastAPI và Telegram</p>
                            <p class="summary-item-text">Nhận câu hỏi, gọi pipeline RAG và trả phản hồi cho người dùng.</p>
                        </div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-icon"><i class="fa-solid fa-database"></i></div>
                        <div>
                            <p class="summary-item-title">ChromaDB</p>
                            <p class="summary-item-text">Lưu tri thức đã được vector hóa để truy xuất các đoạn liên quan.</p>
                        </div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-icon"><i class="fa-solid fa-brain"></i></div>
                        <div>
                            <p class="summary-item-title">OpenRouter và LLM</p>
                            <p class="summary-item-text">Tạo câu trả lời cuối cùng dựa trên ngữ cảnh mà hệ thống truy xuất được.</p>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            f"""
            <div class="surface-card">
                <h3 class="section-title">Trạng thái và cấu hình đang dùng</h3>
                <p class="section-caption">Ưu tiên phần kết nối và cấu hình thực tế để trình bày ngắn gọn, rõ ràng.</p>
                <div class="summary-list">
                    <div class="summary-item">
                        <div class="summary-item-icon"><i class="fa-solid fa-bolt"></i></div>
                        <div>
                            <p class="summary-item-title">OpenRouter API</p>
                            <p class="summary-item-text">{escape(openrouter_status["label"])} · {escape(openrouter_status["detail"])}</p>
                        </div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-icon"><i class="fa-solid fa-boxes-stacked"></i></div>
                        <div>
                            <p class="summary-item-title">ChromaDB</p>
                            <p class="summary-item-text">{escape(chroma_status["label"])} · {escape(chroma_status["detail"])}</p>
                        </div>
                    </div>
                </div>
                <div class="chip-row" style="margin-top:0.9rem;">
                    <span class="chip"><i class="fa-solid fa-brain"></i>{escape(settings_snapshot["openrouter_model"])}</span>
                    <span class="chip"><i class="fa-solid fa-database"></i>{escape(settings_snapshot["chroma_collection"])}</span>
                    <span class="chip"><i class="fa-solid fa-microchip"></i>{escape(settings_snapshot["embedding_device"])}</span>
                    <span class="chip"><i class="fa-solid fa-user-group"></i>{metrics["unique_users"]} người dùng</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_knowledge_page(
    pdf_inventory: list[dict[str, Any]],
    settings_snapshot: dict[str, str],
    chroma_status: dict[str, Any],
) -> None:
    """Render the knowledge management page."""
    render_header(
        "fa-solid fa-folder-tree",
        "Knowledge Base",
        "Quản lý tri thức",
        "Thêm tài liệu PDF mới vào hệ thống, theo dõi kho tri thức hiện có và chạy indexing để cập nhật ChromaDB phục vụ truy xuất RAG.",
    )

    st.session_state.setdefault("knowledge_logs", [])
    st.session_state.setdefault("knowledge_summary", None)
    st.session_state.setdefault("knowledge_last_uploaded", "")
    st.session_state.setdefault("knowledge_flash", "")

    cards = st.columns(3)
    total_storage = sum(item["size_bytes"] for item in pdf_inventory)
    with cards[0]:
        render_stat_card("Số lượng PDF", str(len(pdf_inventory)), "Tổng số tệp PDF trong data.", "fa-regular fa-file-pdf")
    with cards[1]:
        render_stat_card("Dung lượng dữ liệu", format_bytes(total_storage), "Dung lượng cộng dồn của các PDF.", "fa-solid fa-hard-drive")
    with cards[2]:
        render_stat_card(
            "Vectors hiện có",
            str(chroma_status.get("vector_count", 0)),
            "Tổng số vector trong collection hiện tại.",
            "fa-solid fa-database",
        )

    if st.session_state["knowledge_flash"]:
        render_notice("success", "Upload thành công", st.session_state["knowledge_flash"])
        st.session_state["knowledge_flash"] = ""

    start_indexing = False

    st.write("")
    upload_col, action_col = st.columns(2, gap="large")
    with upload_col:
        with st.container(border=True):
            st.markdown("### Upload PDF mới")
            st.caption("Kéo thả hoặc chọn tệp, sau đó lưu trực tiếp vào thư mục `data/`.")
            uploaded_file = st.file_uploader(
                "Kéo thả hoặc chọn tệp PDF",
                type=["pdf"],
                help="Tệp sẽ được lưu trực tiếp vào thư mục data/.",
            )
            if uploaded_file is not None and st.button("Lưu vào thư mục data", width="stretch", type="primary"):
                target_path = Path("data") / uploaded_file.name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(uploaded_file.getbuffer())
                st.session_state["knowledge_last_uploaded"] = uploaded_file.name
                st.session_state["knowledge_flash"] = f"Đã lưu tệp {uploaded_file.name} vào thư mục data."
                st.rerun()

            if st.session_state["knowledge_last_uploaded"]:
                st.markdown(
                    f"""
                    <div class="chip-row">
                        <span class="chip"><i class="fa-solid fa-check"></i>Tệp gần nhất: {escape(st.session_state["knowledge_last_uploaded"])}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with action_col:
        with st.container(border=True):
            st.markdown("### Bắt đầu indexing")
            st.caption("Chạy indexing sau khi upload xong để cập nhật collection trong ChromaDB.")
            st.markdown(
                f"""
                <div class="chip-row">
                    <span class="chip"><i class="fa-solid fa-boxes-stacked"></i>{escape(settings_snapshot["chroma_collection"])}</span>
                    <span class="chip"><i class="fa-solid fa-location-dot"></i>{escape(settings_snapshot["chroma_persist_dir"])}</span>
                    <span class="chip"><i class="fa-solid fa-brain"></i>{escape(settings_snapshot["embedding_model"])}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            has_pdf = any(Path("data").glob("*.pdf"))
            start_indexing = st.button(
                "Bắt đầu Indexing",
                width="stretch",
                type="primary",
                disabled=not has_pdf,
            )
            if not has_pdf:
                render_notice("warning", "Chưa có dữ liệu", "Cần có ít nhất một file PDF trong thư mục data trước khi chạy indexing.")

    st.write("")
    table_col, log_col = st.columns([1.02, 0.98], gap="large")
    with table_col:
        with st.container(border=True):
            st.markdown("### Danh sách PDF hiện có")
            if pdf_inventory:
                rows = [
                    {
                        "Tên file": item["name"],
                        "Dung lượng": format_bytes(item["size_bytes"]),
                        "Cập nhật": item["modified_at"].strftime("%d/%m/%Y %H:%M"),
                    }
                    for item in pdf_inventory
                ]
                st.dataframe(rows, width="stretch", hide_index=True)
            else:
                st.markdown(
                    '<div class="empty-state">Chưa có tài liệu PDF nào trong thư mục data.</div>',
                    unsafe_allow_html=True,
                )

    with log_col:
        with st.container(border=True):
            st.markdown("### Tiến trình và log")
            if start_indexing:
                run_indexing_workflow(settings_snapshot)

            if st.session_state["knowledge_summary"]:
                summary = st.session_state["knowledge_summary"]
                st.markdown(
                    f"""
                    <div class="chip-row" style="margin-bottom:0.85rem;">
                        <span class="chip"><i class="fa-solid fa-folder-open"></i>{summary["files_indexed"]} files</span>
                        <span class="chip"><i class="fa-solid fa-layer-group"></i>{summary["chunks_indexed"]} chunks</span>
                        <span class="chip"><i class="fa-solid fa-database"></i>{summary["vector_count"]} vectors</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if st.session_state["knowledge_logs"]:
                st.code("\n".join(st.session_state["knowledge_logs"][-180:]), language="text")
            else:
                st.markdown(
                    '<div class="empty-state">Log indexing sẽ hiển thị tại đây sau khi bạn bấm nút bắt đầu indexing.</div>',
                    unsafe_allow_html=True,
                )


def render_history_page(records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    """Render the conversation history page."""
    render_header(
        "fa-solid fa-timeline",
        "Conversation History",
        "Lịch sử hội thoại",
        "Theo dõi các câu hỏi đã đi qua chatbot để kiểm tra chất lượng vận hành, độ trễ phản hồi và mức độ ổn định của hệ thống.",
    )

    cards = st.columns(3)
    with cards[0]:
        render_stat_card("Lượt thành công", str(metrics["successful_questions"]), "Số yêu cầu được bot xử lý ổn định.", "fa-solid fa-circle-check")
    with cards[1]:
        render_stat_card("Lượt lỗi", str(metrics["failed_questions"]), "Những phiên cần kiểm tra thêm.", "fa-solid fa-triangle-exclamation")
    with cards[2]:
        render_stat_card("Người dùng duy nhất", str(metrics["unique_users"]), "Tính theo user_id Telegram.", "fa-solid fa-user-group")

    st.write("")
    if not records:
        render_notice("info", "Chưa có dữ liệu hội thoại", "Bot chưa ghi nhận phiên trò chuyện nào trong file log hiện tại.")
        return

    history_rows = [
        {
            "Thời gian": record.get("timestamp", ""),
            "Người dùng": record.get("full_name") or record.get("username") or "Ẩn danh",
            "Câu hỏi": record.get("question", ""),
            "Phản hồi (ms)": record.get("response_time_ms", 0),
            "Trạng thái": record.get("status", ""),
        }
        for record in records[:60]
    ]
    st.dataframe(history_rows, width="stretch", hide_index=True)

    st.write("")
    st.markdown("### Phiên gần nhất")
    for record in records[:8]:
        user_label = record.get("full_name") or record.get("username") or "Ẩn danh"
        latency = record.get("response_time_ms", 0)
        with st.expander(f'{record.get("timestamp", "")} · {user_label} · {latency} ms', expanded=False):
            st.markdown(
                f"""
                <div class="chunk-card chunk-tone-0">
                    <div class="chunk-head">
                        <div class="chunk-title">
                            <span class="meta-pill"><i class="fa-solid fa-circle-question"></i>Câu hỏi</span>
                        </div>
                    </div>
                    <div class="chunk-content">{escape(record.get("question", "")).replace(chr(10), "<br>")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                f"""
                <div class="chunk-card chunk-tone-1">
                    <div class="chunk-head">
                        <div class="chunk-title">
                            <span class="meta-pill"><i class="fa-solid fa-robot"></i>Trả lời</span>
                            <span class="score-pill"><i class="fa-regular fa-clock"></i>{escape(str(latency))} ms</span>
                        </div>
                    </div>
                    <div class="chunk-content">{escape(record.get("answer", "")).replace(chr(10), "<br>")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_config_page(
    env_values: dict[str, str],
    openrouter_status: dict[str, Any],
    chroma_status: dict[str, Any],
) -> None:
    """Render the AI configuration page."""
    render_header(
        "fa-solid fa-sliders",
        "Runtime Configuration",
        "Cấu hình AI",
        "Quản lý các thông số quan trọng của runtime, kiểm tra kết nối OpenRouter và ChromaDB, đồng thời cập nhật nhanh file .env ngay trên dashboard.",
    )

    status_cols = st.columns(2, gap="large")
    with status_cols[0]:
        st.markdown(
            f"""
            <div class="surface-card">
                <h3 class="section-title">OpenRouter API</h3>
                <p class="section-caption">{escape(openrouter_status["detail"])}</p>
                <div class="chip-row">
                    <span class="chip"><i class="fa-solid fa-bolt"></i>{escape(openrouter_status["label"])}</span>
                    <span class="chip"><i class="fa-solid fa-brain"></i>{escape(env_values.get("OPENROUTER_MODEL", env_values.get("GROQ_MODEL", "meta-llama/llama-3.1-8b-instruct")))}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_cols[1]:
        st.markdown(
            f"""
            <div class="surface-card">
                <h3 class="section-title">ChromaDB</h3>
                <p class="section-caption">{escape(chroma_status["detail"])}</p>
                <div class="chip-row">
                    <span class="chip"><i class="fa-solid fa-database"></i>{escape(chroma_status["label"])}</span>
                    <span class="chip"><i class="fa-solid fa-boxes-stacked"></i>{escape(env_values.get("CHROMA_COLLECTION", "dlu_knowledge"))}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    left_col, right_col = st.columns([0.92, 1.08], gap="large")
    with left_col:
        st.markdown("### Snapshot cấu hình hiện tại")
        snapshot_rows = [
            {"Thông số": "Telegram token", "Giá trị": mask_secret(env_values.get("TOKEN", ""))},
            {"Thông số": "OpenRouter API key", "Giá trị": mask_secret(env_values.get("OPENROUTER_API_KEY", env_values.get("GROQ_API_KEY", "")))},
            {"Thông số": "OpenRouter model", "Giá trị": env_values.get("OPENROUTER_MODEL", env_values.get("GROQ_MODEL", "meta-llama/llama-3.1-8b-instruct"))},
            {"Thông số": "OpenRouter site URL", "Giá trị": env_values.get("OPENROUTER_SITE_URL", "")},
            {"Thông số": "OpenRouter site name", "Giá trị": env_values.get("OPENROUTER_SITE_NAME", "DLU Chatbot")},
            {"Thông số": "Chroma collection", "Giá trị": env_values.get("CHROMA_COLLECTION", "dlu_knowledge")},
            {"Thông số": "Persist dir", "Giá trị": env_values.get("CHROMA_PERSIST_DIR", "vector_store")},
            {"Thông số": "Embedding model", "Giá trị": env_values.get("EMBEDDING_MODEL", "")},
            {"Thông số": "Embedding device", "Giá trị": env_values.get("EMBEDDING_DEVICE", "cpu")},
        ]
        st.dataframe(snapshot_rows, width="stretch", hide_index=True)

    with right_col:
        st.markdown("### Cập nhật nhanh file `.env`")
        with st.form("config_form"):
            openrouter_model = st.text_input("OpenRouter model", value=env_values.get("OPENROUTER_MODEL", env_values.get("GROQ_MODEL", "meta-llama/llama-3.1-8b-instruct")))
            openrouter_site_url = st.text_input("OpenRouter site URL", value=env_values.get("OPENROUTER_SITE_URL", ""))
            openrouter_site_name = st.text_input("OpenRouter site name", value=env_values.get("OPENROUTER_SITE_NAME", "DLU Chatbot"))
            chroma_collection = st.text_input("Chroma collection", value=env_values.get("CHROMA_COLLECTION", "dlu_knowledge"))
            chroma_persist_dir = st.text_input("Chroma persist dir", value=env_values.get("CHROMA_PERSIST_DIR", "vector_store"))
            embedding_model = st.text_input(
                "Embedding model",
                value=env_values.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            )
            embedding_device = st.selectbox(
                "Embedding device",
                options=["cpu", "cuda"],
                index=0 if env_values.get("EMBEDDING_DEVICE", "cpu") == "cpu" else 1,
            )
            webhook_url = st.text_input("Webhook URL", value=env_values.get("WEBHOOK_URL", ""))
            new_openrouter_api_key = st.text_input("OpenRouter API key mới", value="", type="password")
            new_token = st.text_input("Telegram token mới", value="", type="password")
            save_button = st.form_submit_button("Lưu cấu hình", width="stretch", type="primary")

        if save_button:
            updates = {
                "OPENROUTER_MODEL": openrouter_model.strip(),
                "OPENROUTER_SITE_URL": openrouter_site_url.strip(),
                "OPENROUTER_SITE_NAME": openrouter_site_name.strip(),
                "CHROMA_COLLECTION": chroma_collection.strip(),
                "CHROMA_PERSIST_DIR": chroma_persist_dir.strip(),
                "EMBEDDING_MODEL": embedding_model.strip(),
                "EMBEDDING_DEVICE": embedding_device.strip(),
                "WEBHOOK_URL": webhook_url.strip(),
            }
            if new_openrouter_api_key.strip():
                updates["OPENROUTER_API_KEY"] = new_openrouter_api_key.strip()
            if new_token.strip():
                updates["TOKEN"] = new_token.strip()

            update_env_values(updates)
            clear_runtime_caches()
            render_notice("success", "Đã cập nhật cấu hình", "File .env đã được lưu. Nếu backend đang chạy, hãy khởi động lại service để áp dụng thay đổi.")


def render_chunk_block(chunk: dict[str, Any], tone_index: int) -> None:
    """Render one retrieval chunk in a styled card."""
    tone_class = f"chunk-tone-{tone_index % 4}"
    metadata_pills = []
    if chunk.get("page") not in ("", "unknown", None):
        metadata_pills.append(f'<span class="meta-pill"><i class="fa-regular fa-file-lines"></i>Trang {escape(str(chunk["page"]))}</span>')
    if chunk.get("section_title"):
        metadata_pills.append(
            f'<span class="meta-pill"><i class="fa-solid fa-bookmark"></i>{escape(str(chunk["section_title"]))}</span>'
        )

    st.markdown(
        f"""
        <div class="chunk-card {tone_class}">
            <div class="chunk-head">
                <div class="chunk-title">
                    <span class="file-pill"><i class="fa-regular fa-file-pdf"></i>{escape(chunk["source"])}</span>
                    <span class="score-pill"><i class="fa-solid fa-wave-square"></i>{chunk["score_percent"]:.1f}% similarity</span>
                </div>
                <div class="chunk-title">{"".join(metadata_pills)}</div>
            </div>
            <div class="chunk-content">{escape(chunk["content"]).replace(chr(10), "<br>")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_inspector_page(settings_snapshot: dict[str, str], collection_rows: list[dict[str, Any]]) -> None:
    """Render the RAG inspection page."""
    render_header(
        "fa-solid fa-magnifying-glass-chart",
        "RAG Debug View",
        "RAG Inspector",
        "Quan sát trực quan toàn bộ quy trình RAG: câu hỏi đầu vào, các chunks được truy xuất từ ChromaDB với similarity score, và câu trả lời cuối cùng của Llama 3.",
    )

    st.session_state.setdefault("rag_result", None)
    st.session_state.setdefault("rag_error", "")
    st.session_state.setdefault("rag_last_query", "Sinh viên xem lịch học ở đâu?")

    collection_lookup = {item["name"]: item["vector_count"] for item in collection_rows}
    collection_names = [item["name"] for item in collection_rows] or [settings_snapshot["chroma_collection"]]
    default_collection = settings_snapshot["chroma_collection"]
    default_index = collection_names.index(default_collection) if default_collection in collection_names else 0

    if collection_lookup.get(default_collection, 0) == 0:
        suggestion = next((name for name in collection_names if collection_lookup.get(name, 0) > 0), None)
        if suggestion:
            render_notice(
                "warning",
                "Collection runtime đang rỗng",
                f"Collection {default_collection} hiện chưa có vector. Bạn có thể chọn {suggestion} để demo retrieval rõ hơn.",
            )

    with st.form("rag_form"):
        query = st.text_area(
            "Câu hỏi kiểm thử",
            value=st.session_state["rag_last_query"],
            height=120,
            placeholder="Ví dụ: Điều kiện xét học bổng khuyến khích học tập là gì?",
        )
        form_cols = st.columns([1.05, 0.9, 0.8], gap="large")
        with form_cols[0]:
            selected_collection = st.selectbox("Collection", options=collection_names, index=default_index)
        with form_cols[1]:
            top_k = st.slider("Số chunk truy xuất", min_value=1, max_value=5, value=3)
        with form_cols[2]:
            st.write("")
            st.write("")
            analyze = st.form_submit_button("Phân tích truy xuất", width="stretch", type="primary")

    if analyze:
        normalized_query = query or ""
        st.session_state["rag_last_query"] = normalized_query
        st.session_state["rag_error"] = ""
        with st.spinner("Đang truy xuất ChromaDB và gọi OpenRouter..."):
            try:
                st.session_state["rag_result"] = run_async_task(
                    inspect_rag_pipeline(
                        normalized_query,
                        top_k=top_k,
                        collection_name=selected_collection,
                    )
                )
            except Exception as exc:
                st.session_state["rag_result"] = None
                st.session_state["rag_error"] = str(exc)

    if st.session_state["rag_error"]:
        render_notice("error", "Không thể chạy RAG Inspector", st.session_state["rag_error"])
        return

    result = st.session_state["rag_result"]
    if not result:
        render_notice("info", "Chưa có kết quả inspect", "Nhập một câu hỏi kiểm thử rồi bấm nút phân tích để xem toàn bộ luồng RAG.")
        return

    chunks = result["chunks"]
    best_score = max((chunk["score_percent"] for chunk in chunks), default=0.0)
    stat_cols = st.columns(3)
    with stat_cols[0]:
        render_stat_card("Collection đang dùng", result["collection_name"], "Nguồn vector cho lượt kiểm thử hiện tại.", "fa-solid fa-boxes-stacked")
    with stat_cols[1]:
        render_stat_card("Chunks truy xuất", str(len(chunks)), "Top-K thực tế lấy về từ ChromaDB.", "fa-solid fa-layer-group")
    with stat_cols[2]:
        render_stat_card("Similarity cao nhất", f"{best_score:.1f}%", "Chunk gần câu hỏi nhất trong lượt truy xuất.", "fa-solid fa-wave-square")

    st.write("")
    with st.container(border=True):
        st.markdown("### 1. Câu hỏi người dùng")
        st.caption("Đầu vào được embedding và dùng để truy xuất context.")
        st.markdown(
            f"""
            <div class="chunk-card chunk-tone-0">
                <div class="chunk-content">{escape(result["query"]).replace(chr(10), "<br>")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown("### 2. Chunks truy xuất từ ChromaDB")
        st.caption("Các đoạn văn bản được hiển thị trong các khung màu nền khác nhau, kèm nhãn file PDF và similarity score.")
        st.markdown(
            f"""
            <div class="chip-row" style="margin-bottom:0.8rem;">
                <span class="chip"><i class="fa-solid fa-database"></i>{escape(result["collection_name"])}</span>
                <span class="chip"><i class="fa-solid fa-brain"></i>{escape(result["model"])}</span>
                <span class="chip"><i class="fa-solid fa-layer-group"></i>{len(chunks)} chunks</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if chunks:
            for index, chunk in enumerate(chunks):
                with st.expander(
                    f"Chunk {chunk['rank']} · {chunk['source']} · Similarity {chunk['score_percent']:.1f}%",
                    expanded=index == 0,
                ):
                    render_chunk_block(chunk, tone_index=index)
        else:
            render_notice("warning", "Không có chunk nào", "Collection đã chọn không trả về context phù hợp cho câu hỏi hiện tại.")

    with st.container(border=True):
        st.markdown("### 3. Câu trả lời cuối cùng từ Llama 3")
        st.caption("Output cuối cùng sau khi model đọc các chunks được truy xuất.")
        st.markdown(
            f"""
            <div class="chunk-card chunk-tone-1">
                <div class="chunk-content">{escape(result["answer"]).replace(chr(10), "<br>")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    """Run the Streamlit dashboard application."""
    inject_styles()

    env_values = load_env_values()
    settings_snapshot = load_settings_snapshot()
    records = load_conversation_history()
    metrics = compute_dashboard_metrics(records)
    pdf_inventory = get_pdf_inventory()
    openrouter_status = check_openrouter_connection(
        env_values.get("OPENROUTER_API_KEY", env_values.get("GROQ_API_KEY")),
        settings_snapshot["openrouter_model"],
        site_url=settings_snapshot["openrouter_site_url"],
        site_name=settings_snapshot["openrouter_site_name"],
    )
    chroma_status = check_chromadb_connection(
        collection_name=settings_snapshot["chroma_collection"],
        persist_dir=settings_snapshot["chroma_persist_dir"],
    )
    collection_rows = list_chroma_collections(settings_snapshot["chroma_persist_dir"])

    active_page = render_sidebar(
        metrics=metrics,
        pdf_count=len(pdf_inventory),
        collection_count=len(collection_rows),
    )

    if active_page == "overview":
        render_overview_page(
            metrics=metrics,
            pdf_inventory=pdf_inventory,
            openrouter_status=openrouter_status,
            chroma_status=chroma_status,
            settings_snapshot=settings_snapshot,
        )
    elif active_page == "inspector":
        render_inspector_page(settings_snapshot=settings_snapshot, collection_rows=collection_rows)
    elif active_page == "knowledge":
        render_knowledge_page(
            pdf_inventory=pdf_inventory,
            settings_snapshot=settings_snapshot,
            chroma_status=chroma_status,
        )
    elif active_page == "history":
        render_history_page(records=records, metrics=metrics)
    else:
        render_config_page(
            env_values=env_values,
            openrouter_status=openrouter_status,
            chroma_status=chroma_status,
        )


if __name__ == "__main__":
    main()
