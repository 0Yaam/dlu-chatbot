from __future__ import annotations

import argparse
import re
from pathlib import Path

import chromadb
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class DLUKnowledgeIndexer:
    """Ingest DLU documents and store chunks in a local ChromaDB collection."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        persist_dir: str | Path = "vector_store",
        collection_name: str = "dlu_knowledge",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        device: str = "cpu",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.device,
        )

    def run(self) -> None:
        """Load, chunk, embed, and store all supported documents."""
        print("=" * 80)
        print("BAT DAU INDEX DU LIEU CHO DLU CHATBOT")
        print(f"Thu muc data: {self.data_dir.resolve()}")
        print(f"Thu muc vector_store: {self.persist_dir.resolve()}")
        print(f"Collection: {self.collection_name}")
        print(f"Embedding model: {self.embedding_model_name}")
        print("=" * 80)

        documents = self.load_documents()
        if not documents:
            print("Khong co document hop le nao de index.")
            return

        self.create_vector_db(documents)
        print(f"Hoan tat. Tong so chunk da dua vao ChromaDB: {len(documents)}")

    def load_documents(self) -> list[Document]:
        """Process all supported files in the raw data directory."""
        if not self.data_dir.exists():
            print(f"Khong tim thay thu muc data: {self.data_dir}")
            return []

        all_chunks: list[Document] = []
        supported_files = sorted(
            path
            for path in self.data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}
        )

        if not supported_files:
            print(f"Thu muc {self.data_dir} khong co file .pdf hoac .txt nao.")
            return []

        for file_path in supported_files:
            try:
                print(f"\nDang xu ly file: {file_path.name}")
                file_chunks = self.process_file(file_path)
                print(f"Da tao {len(file_chunks)} chunk tu file {file_path.name}")
                all_chunks.extend(file_chunks)
            except Exception as exc:
                print(f"Loi khi xu ly file {file_path.name}: {exc}")

        print(f"\nTong so chunk sau khi xu ly tat ca file: {len(all_chunks)}")
        return all_chunks

    def process_file(self, file_path: Path) -> list[Document]:
        """Dispatch processing based on file extension and filename."""
        suffix = file_path.suffix.lower()

        if suffix == ".txt" and file_path.name.lower() == "faq.txt":
            return self.process_faq_file(file_path)
        if suffix == ".pdf":
            return self.process_pdf_file(file_path)
        if suffix == ".txt":
            return self.process_text_file(file_path)

        print(f"Bo qua file khong ho tro: {file_path.name}")
        return []

    def process_faq_file(self, file_path: Path) -> list[Document]:
        """Split FAQ into question-answer blocks, fallback to recursive splitting when needed."""
        raw_text = self.read_text_file(file_path)
        if not raw_text.strip():
            print(f"File {file_path.name} khong co noi dung.")
            return []

        faq_blocks = [block.strip() for block in re.split(r"\n\s*\n", raw_text) if block.strip()]
        documents: list[Document] = []

        for block_index, block in enumerate(faq_blocks, start=1):
            metadata = {
                "source": file_path.name,
                "source_type": "faq",
                "block_index": block_index,
            }

            if len(block) <= self.chunk_size:
                documents.append(Document(page_content=block, metadata=metadata | {"chunk_method": "faq_block"}))
            else:
                documents.extend(self.chunk_text(block, metadata, method="faq_recursive"))

        return documents

    def process_text_file(self, file_path: Path) -> list[Document]:
        """Process generic text files with recursive splitting."""
        raw_text = self.read_text_file(file_path)
        if not raw_text.strip():
            print(f"File {file_path.name} khong co noi dung.")
            return []

        return self.chunk_text(
            text=raw_text,
            metadata={"source": file_path.name, "source_type": "text"},
            method="text_recursive",
        )

    def process_pdf_file(self, file_path: Path) -> list[Document]:
        """Extract PDF text, try splitting by 'Dieu', then fallback to recursive splitting."""
        full_text = self.extract_pdf_text(file_path)
        if not full_text.strip():
            print(f"Khong doc duoc noi dung tu file PDF: {file_path.name}")
            return []

        article_sections = self.split_by_dieu(full_text)
        documents: list[Document] = []

        if not article_sections:
            print(f"Khong tim thay cac muc 'Dieu' trong {file_path.name}. Fallback sang recursive split.")
            return self.chunk_text(
                text=full_text,
                metadata={"source": file_path.name, "source_type": "pdf"},
                method="pdf_recursive_full",
            )

        for section_index, section in enumerate(article_sections, start=1):
            section_title = self.extract_first_line(section)
            metadata = {
                "source": file_path.name,
                "source_type": "pdf",
                "section_index": section_index,
                "section_title": section_title,
            }

            if len(section) <= self.chunk_size:
                documents.append(
                    Document(
                        page_content=section,
                        metadata=metadata | {"chunk_method": "pdf_dieu_regex"},
                    )
                )
            else:
                documents.extend(self.chunk_text(section, metadata, method="pdf_recursive_section"))

        return documents

    def extract_pdf_text(self, file_path: Path) -> str:
        """Read PDF text with pdfplumber and normalize whitespace."""
        pages: list[str] = []

        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = (page.extract_text() or "").strip()
                    if page_text:
                        normalized = self.normalize_text(page_text)
                        pages.append(normalized)
                    else:
                        print(f"Trang {page_number} cua {file_path.name} khong co text.")
                except Exception as exc:
                    print(f"Loi khi doc trang {page_number} cua {file_path.name}: {exc}")

        return "\n\n".join(pages)

    def split_by_dieu(self, text: str) -> list[str]:
        """Split text by article markers such as 'Dieu 1.', 'Dieu 2.'."""
        normalized_text = self.normalize_text(text)
        pattern = re.compile(r"(?im)^\s*(?:Dieu|\u0110i\u1ec1u)\s+\d+[^\n]*")
        matches = list(pattern.finditer(normalized_text))

        if not matches:
            return []

        sections: list[str] = []

        intro_text = normalized_text[: matches[0].start()].strip()
        if intro_text:
            sections.append(intro_text)

        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
            section = normalized_text[start:end].strip()
            if section:
                sections.append(section)

        return sections

    def chunk_text(self, text: str, metadata: dict, method: str) -> list[Document]:
        """Fallback chunking using RecursiveCharacterTextSplitter."""
        clean_text = self.normalize_text(text)
        if not clean_text:
            return []

        chunks = self.splitter.create_documents([clean_text], metadatas=[metadata | {"chunk_method": method}])
        return chunks

    def create_vector_db(self, documents: list[Document], batch_size: int = 32) -> None:
        """Embed documents locally and upsert them into ChromaDB."""
        if not documents:
            print("Khong co chunk nao de luu vao ChromaDB.")
            return

        print("\nBat dau tao embedding va luu vao ChromaDB...")
        for batch_start in range(0, len(documents), batch_size):
            batch_docs = documents[batch_start : batch_start + batch_size]
            batch_ids = [self.build_document_id(doc, batch_start + offset) for offset, doc in enumerate(batch_docs)]
            batch_texts = [doc.page_content for doc in batch_docs]
            batch_metadatas = [self.sanitize_metadata(doc.metadata) for doc in batch_docs]

            try:
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()

                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    embeddings=batch_embeddings,
                )
                print(
                    f"Da luu batch {batch_start + 1}-{batch_start + len(batch_docs)} / {len(documents)} vao ChromaDB"
                )
            except Exception as exc:
                print(f"Loi khi luu batch {batch_start + 1}-{batch_start + len(batch_docs)}: {exc}")

        total_in_collection = self.collection.count()
        print(f"Collection '{self.collection_name}' hien co tong cong {total_in_collection} vector.")

    @staticmethod
    def read_text_file(file_path: Path) -> str:
        """Read text file with a few common UTF-8 variants."""
        encodings = ("utf-8", "utf-8-sig")
        last_error: Exception | None = None

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        raise ValueError(f"Khong doc duoc file text: {file_path}")

    @staticmethod
    def normalize_text(text: str) -> str:
        """Clean repeated whitespace while preserving paragraph boundaries."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def extract_first_line(text: str) -> str:
        """Return the first non-empty line as a lightweight section title."""
        for line in text.splitlines():
            clean_line = line.strip()
            if clean_line:
                return clean_line[:200]
        return ""

    @staticmethod
    def sanitize_metadata(metadata: dict) -> dict:
        """Convert metadata to Chroma-friendly primitive values."""
        clean_metadata: dict = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            else:
                clean_metadata[key] = str(value)
        return clean_metadata

    @staticmethod
    def build_document_id(document: Document, sequence_number: int) -> str:
        """Generate a stable-enough id for Chroma upsert."""
        source = str(document.metadata.get("source", "unknown")).replace(" ", "_")
        chunk_method = str(document.metadata.get("chunk_method", "chunk"))
        start_index = document.metadata.get("start_index", sequence_number)
        return f"{source}-{chunk_method}-{start_index}-{sequence_number}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index DLU raw documents into local ChromaDB.")
    parser.add_argument("--data-dir", default="data", help="Thu muc chua file dau vao.")
    parser.add_argument("--persist-dir", default="vector_store", help="Thu muc luu ChromaDB.")
    parser.add_argument("--collection-name", default="dlu_knowledge", help="Ten collection trong ChromaDB.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Ten model embedding local.",
    )
    parser.add_argument("--device", default="cpu", help="Thiet bi chay embedding model, vi du: cpu hoac cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indexer = DLUKnowledgeIndexer(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
        device=args.device,
    )

    try:
        indexer.run()
    except Exception as exc:
        print(f"Loi nghiem trong khi index du lieu: {exc}")


if __name__ == "__main__":
    main()
