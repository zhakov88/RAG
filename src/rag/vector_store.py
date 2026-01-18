import logging
import os
import uuid
from typing import Any, Dict, List

import chromadb
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages storage of document embeddings using ChromaDB."""

    def __init__(
        self,
        collection_name: str,
        persist_dir: str,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF Document Embeddings for RAG"},
            )
            logger.info(
                f"Vector store initialized with collection: {self.collection_name}"
            )
            logger.info(f"Existing documents in store: {self.collection.count()}")
        except Exception as e:
            logger.warning(f"Error initializing vector store: {e}")
            raise e

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector store."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")

        # Prepare data for chromadb
        ids = []
        metadata_list = []
        documents_texts = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate a unique ID for each document
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata, text, and embedding
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadata_list.append(metadata)

            # Document content
            documents_texts.append(doc.page_content)

            # Embedding
            embeddings_list.append(embedding.tolist())

        # Add to chromadb collection
        try:
            self.collection.add(
                ids=ids,
                metadatas=metadata_list,
                documents=documents_texts,
                embeddings=embeddings_list,
            )
            print(f"Added {len(documents)} documents to the vector store.")
            print(f"Total documents in store: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5):
        """Retrieve top_k documents similar to the query embedding."""
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
            )
            return results
        except Exception as e:
            logger.warning(f"Error retrieving documents: {e}")
            return None


# Example usage
if __name__ == "__main__":
    from rag.config import COLLECTION_NAME, PERSIST_DIRECTORY
    from rag.logging_config import setup_logging

    setup_logging()

    # Example usage
    vector_store = VectorStore(
        collection_name=COLLECTION_NAME,
        persist_dir=PERSIST_DIRECTORY,
    )
    print(vector_store)
