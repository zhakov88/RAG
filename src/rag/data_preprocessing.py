import logging
from pathlib import Path
from typing import Any, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingTransformer:
    """Handles embedding generation using SentenceTransformer."""

    def __init__(
        self,
        model_name: str,
    ):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(
                f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
            )
        except Exception as e:
            logger.info(f"Error loading model {self.model_name}: {e}")
            raise e

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
        logger.info(f"Generating embeddings for {len(texts)} texts.")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


class DataPipeline:
    """Pipeline to load, split, and embed documents."""

    def __init__(
        self,
        data_dir: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model_name: str,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name

    def run(self):
        # Load documents
        documents = self.load_all_documents()
        logger.info(f"Loaded {len(documents)} documents from {self.data_dir}.")

        # Split documents
        split_docs = self.split_documents(documents)

        # Generate embeddings
        embedder = EmbeddingTransformer(model_name=self.embedding_model_name)
        texts = [doc.page_content for doc in split_docs]
        embeddings = embedder.generate_embeddings(texts)
        logger.info(f"Generated embeddings for {len(split_docs)} chunks.")

        return split_docs, embeddings

    def load_all_documents(self) -> List[Any]:
        """
        Load all documents from the specified directory using appropriate loaders
        based on file extensions.

        Args:
            directory_path (str): The path to the directory containing documents.

        Returns:
            List[Any]: A list of loaded documents.
        """

        # Use project root data folder
        data_path = Path(self.data_dir).resolve()
        logger.debug(f"Data path: {data_path}")

        documents = []

        # PDF files
        pdf_files = list(data_path.glob("*.pdf"))
        logger.debug(
            f"Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}."
        )
        for pdf_file in pdf_files:
            logger.debug(f"Loading PDF: {pdf_file}.")
            try:
                loader = PyPDFLoader(str(pdf_file))
                loaded_docs = loader.load()
                logger.debug(f"Loaded {len(loaded_docs)} documents from {pdf_file}.")

                # Add source metadata to each document
                for doc in loaded_docs:
                    doc.metadata["source_file"] = pdf_file.name

                documents.extend(loaded_docs)
            except Exception as e:
                logger.debug(f"[ERROR] Failed to load {pdf_file}: {e}")
        return documents

    def split_documents(self, documents) -> List[Any]:
        """Split documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks.")
        # Show example of chunk
        if split_docs:
            logger.info(
                f"Example chunk: \n\tContent: {split_docs[0].page_content[:200]}...\n\tMetadata: {split_docs[0].metadata}"
            )
        return split_docs

# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    data_pipeline = DataPipeline(
        data_dir="data/pdf",
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model_name="all-MiniLM-L6-v2",
    )
    split_docs, embeddings = data_pipeline.run()


# %%
