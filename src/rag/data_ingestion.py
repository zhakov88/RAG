from rag.data_preprocessing import DataPipeline
from rag.vector_store import VectorStore


class DataIngestion:
    def __init__(
        self,
        data_source: str,
        vector_store: VectorStore,
        chunk_size: int,
        chunk_overlap,
        embedding_model_name: str,
    ):
        self.data_source = data_source
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name

    def ingest_data(self):
        pipeline = DataPipeline(
            data_dir=self.data_source,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_model_name=self.embedding_model_name,
        )
        chunks, embeddings = pipeline.run()
        self.vector_store.add_documents(chunks, embeddings)
        return


# Example usage
if __name__ == "__main__":
    from rag.vector_store import VectorStore
    from rag.logging_config import setup_logging

    setup_logging()

    data_source = "data/pdf"
    collection_name = "my_pdf_collection"
    persist_dir = "data/vector_store"
    vector_store = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
    chunk_size = 1000
    chunk_overlap = 200
    embedding_model_name = "all-MiniLM-L6-v2"

    ingestion = DataIngestion(
        data_source, vector_store, chunk_size, chunk_overlap, embedding_model_name
    )
    ingestion.ingest_data()
