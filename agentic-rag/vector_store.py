"""
Vector Store Manager for Agentic RAG
Manages ChromaDB collections with HuggingFace embeddings.
"""

import os
from typing import List, Tuple, Optional
from langchain_core.documents import Document


class VectorStoreManager:
    """Manage ChromaDB vector store with HuggingFace embeddings."""

    def __init__(self, persist_dir="./chroma_db", collection_name="agentic_rag"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._vectorstore = None
        self._embeddings = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            from langchain_chroma import Chroma
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to the vector store. Returns number of docs added."""
        if not documents:
            return 0
        self.vectorstore.add_documents(documents)
        return len(documents)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_scores(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Search with relevance scores (lower = more similar for distance)."""
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int = 4, score_threshold: Optional[float] = None):
        """Get a LangChain retriever from the vector store."""
        search_kwargs = {"k": k}
        if score_threshold is not None:
            return self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": score_threshold, "k": k},
            )
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store collection."""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            # Get unique sources
            if count > 0:
                results = collection.peek(min(count, 10))
                sources = set()
                if results and "metadatas" in results:
                    for meta in results["metadatas"]:
                        if meta and "source" in meta:
                            sources.add(meta["source"])
                return {
                    "total_chunks": count,
                    "sample_sources": list(sources),
                    "collection_name": self.collection_name,
                }
            return {
                "total_chunks": 0,
                "sample_sources": [],
                "collection_name": self.collection_name,
            }
        except Exception:
            return {
                "total_chunks": 0,
                "sample_sources": [],
                "collection_name": self.collection_name,
            }

    def clear_collection(self):
        """Delete all documents from the collection."""
        try:
            from langchain_chroma import Chroma
            if os.path.exists(self.persist_dir):
                import shutil
                shutil.rmtree(self.persist_dir)
            self._vectorstore = None
        except Exception as e:
            raise RuntimeError(f"Failed to clear collection: {e}")

    def has_documents(self) -> bool:
        """Check if the vector store has any documents."""
        stats = self.get_collection_stats()
        return stats["total_chunks"] > 0
