from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
import pdfplumber
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a PDF file, split its text into chunks, and store embeddings in ChromaDB."
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF file to ingest.")
    parser.add_argument(
        "--collection",
        default="dlu_documents",
        help="Target Chroma collection name.",
    )
    parser.add_argument(
        "--persist-dir",
        default="vector_store",
        help="Directory where the local Chroma database will be stored.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size for RecursiveCharacterTextSplitter.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Overlap size between chunks.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="HuggingFace sentence-transformers model name.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device passed to the embedding model, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete the existing collection before indexing.",
    )
    return parser.parse_args()


def load_pdf_documents(pdf_path: Path) -> list[Document]:
    """Extract text from each PDF page and convert it to LangChain documents."""
    documents: list[Document] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "page": page_number,
                    },
                )
            )

    if not documents:
        raise ValueError(
            "No extractable text was found in the PDF. If this is a scanned PDF, OCR is needed first."
        )

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split extracted PDF documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(
    persist_dir: Path,
    collection_name: str,
    embedding_model: str,
    device: str,
    reset_collection: bool,
) -> Chroma:
    """Create a local ChromaDB-backed LangChain vector store."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if reset_collection:
        existing_collections = {collection.name for collection in client.list_collections()}
        if collection_name in existing_collections:
            client.delete_collection(name=collection_name)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )

    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def build_ids(chunks: list[Document], pdf_path: Path) -> list[str]:
    """Create stable document ids for Chroma records."""
    ids: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        page = chunk.metadata.get("page", "unknown")
        start_index = chunk.metadata.get("start_index", 0)
        ids.append(f"{pdf_path.stem}-p{page}-s{start_index}-c{index}")

    return ids


def index_pdf_to_chroma(
    pdf_path: Path,
    collection_name: str,
    persist_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    device: str,
    reset_collection: bool,
) -> None:
    """Main ingestion flow: extract, split, embed, and store documents."""
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    documents = load_pdf_documents(pdf_path)
    chunks = split_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    ids = build_ids(chunks, pdf_path)

    vector_store = build_vector_store(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        device=device,
        reset_collection=reset_collection,
    )
    vector_store.add_documents(documents=chunks, ids=ids)

    print(f"Indexed PDF: {pdf_path}")
    print(f"Pages with text: {len(documents)}")
    print(f"Chunks stored: {len(chunks)}")
    print(f"Collection: {collection_name}")
    print(f"Persist directory: {persist_dir}")
    print(f"Embedding model: {embedding_model}")


def main() -> None:
    args = parse_args()
    index_pdf_to_chroma(
        pdf_path=Path(args.pdf),
        collection_name=args.collection,
        persist_dir=Path(args.persist_dir),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        device=args.device,
        reset_collection=args.reset_collection,
    )


if __name__ == "__main__":
    main()
