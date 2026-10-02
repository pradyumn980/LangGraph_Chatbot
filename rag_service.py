import os
import io
import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Global vector store and ingested document registry
_vector_store: Optional[InMemoryVectorStore] = None
_ingested_files: List[dict] = []
_embeddings = None


def get_embeddings():
    """Lazy initialize HuggingFace embeddings."""
    global _embeddings
    if _embeddings is None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found for RAG embeddings")
        _embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=hf_token
        )
    return _embeddings


def get_vector_store() -> InMemoryVectorStore:
    """Retrieve or initialize the in-memory vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = InMemoryVectorStore(get_embeddings())
    return _vector_store


def parse_file_to_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from uploaded PDF, TXT, MD, or CSV files."""
    name_lower = filename.lower()
    if name_lower.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(f"[Page {i+1}]\n{text}")
            return "\n\n".join(pages_text)
        except Exception as e:
            logger.error(f"Error reading PDF {filename}: {e}")
            raise ValueError(f"Could not parse PDF '{filename}': {e}")
    else:
        # Default text/markdown/csv reading
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="replace")


def ingest_document(file_bytes: bytes, filename: str, chunk_size: int = 500, chunk_overlap: int = 50) -> int:
    """
    Parse, chunk, and index an uploaded document into the RAG vector store.
    Returns the number of created chunks.
    """
    raw_text = parse_file_to_text(file_bytes, filename)
    if not raw_text.strip():
        raise ValueError(f"File '{filename}' contains no readable text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    doc = Document(
        page_content=raw_text,
        metadata={"source": filename}
    )
    chunks = splitter.split_documents([doc])

    store = get_vector_store()
    store.add_documents(chunks)

    # Track metadata
    _ingested_files.append({
        "filename": filename,
        "chunks": len(chunks),
        "size_chars": len(raw_text)
    })
    logger.info(f"Ingested '{filename}' into RAG vector store: {len(chunks)} chunks.")
    return len(chunks)


def get_ingested_files_info() -> List[dict]:
    """Return list of currently ingested document summaries."""
    return list(_ingested_files)


def clear_rag_store():
    """Clear all documents from the RAG store."""
    global _vector_store, _ingested_files
    _vector_store = InMemoryVectorStore(get_embeddings())
    _ingested_files = []
    logger.info("Cleared RAG vector store.")


@tool
def query_knowledge_base(query: str) -> str:
    """
    Search and retrieve relevant context and answers from user-uploaded documents (PDFs, text files, notes).
    Always use this tool when the user asks questions about their uploaded files, documents, papers, or specific custom data.
    """
    try:
        store = get_vector_store()
        if not _ingested_files:
            return "No documents have been uploaded to the knowledge base yet. Please upload files in the sidebar first."

        results = store.similarity_search(query, k=3)
        if not results:
            return f"No relevant information found in the uploaded documents for query: '{query}'."

        formatted_chunks = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown file")
            content = doc.page_content.strip()
            formatted_chunks.append(f"[Excerpt {i} from {source}]:\n{content}")

        return "\n\n---\n\n".join(formatted_chunks)
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        return f"Error searching knowledge base: {e}"
