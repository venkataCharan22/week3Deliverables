"""
Document Processor for Agentic RAG
Handles loading, splitting, and preparing documents for the vector store.
"""

import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Process documents for RAG - supports PDF, TXT, and MD files."""

    def __init__(self, chunk_size=800, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def load_file(self, file_path: str) -> List[Document]:
        """Load a single file and return list of Documents."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext in (".txt", ".md", ".markdown"):
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def load_uploaded_file(self, uploaded_file) -> List[Document]:
        """Load a Streamlit UploadedFile object."""
        name = uploaded_file.name
        ext = os.path.splitext(name)[1].lower()

        if ext == ".pdf":
            return self._load_pdf_bytes(uploaded_file.read(), name)
        elif ext in (".txt", ".md", ".markdown"):
            content = uploaded_file.read().decode("utf-8")
            doc = Document(page_content=content, metadata={"source": name})
            return [doc]
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        return self.splitter.split_documents(documents)

    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """Full pipeline: load uploaded file -> split into chunks."""
        docs = self.load_uploaded_file(uploaded_file)
        chunks = self.split_documents(docs)
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        return chunks

    def process_file(self, file_path: str) -> List[Document]:
        """Full pipeline: load file -> split into chunks."""
        docs = self.load_file(file_path)
        chunks = self.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        return chunks

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF file."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": os.path.basename(file_path), "page": i + 1}
                    ))
            return docs
        except ImportError:
            raise ImportError("pypdf required for PDF files. Run: pip install pypdf")

    def _load_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> List[Document]:
        """Load PDF from bytes."""
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(pdf_bytes))
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": i + 1}
                    ))
            return docs
        except ImportError:
            raise ImportError("pypdf required for PDF files. Run: pip install pypdf")

    def _load_text(self, file_path: str) -> List[Document]:
        """Load a text/markdown file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [Document(
            page_content=content,
            metadata={"source": os.path.basename(file_path)}
        )]
