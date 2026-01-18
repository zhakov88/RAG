import logging

from langchain_ollama import ChatOllama

from rag.vector_store import VectorStore
from rag.prompts import PROMPT
from rag.data_preprocessing import EmbeddingTransformer

logger = logging.getLogger(__name__)


class RAGSearch:
    def __init__(
        self,
        vector_store: VectorStore,
        chat_model_name: str,
        embedding_transformer: EmbeddingTransformer,
    ):
        self.vector_store = vector_store
        self.llm = ChatOllama(
            model=chat_model_name, temperature=0, max_tokens=5000, top_p=0.9,
        )
        self.embedding_transformer = embedding_transformer
        logger.info(f"LLM initialized: {chat_model_name}")

    def retrieve_and_generate(self, query: str, top_k: int = 5) -> str:
        query_embedding = self.embedding_transformer.generate_embeddings([query])[0]
        results = self.vector_store.retrieve(query_embedding, top_k=top_k)

        context = []
        for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
            metadata_clean = {key: metadata[key] for key in ["source_file", "page"]}
            context.append({"text": text, "metadata": metadata_clean})

        if not context:
            return "No relevant documents found."
        prompt = PROMPT.format(input=query, context=context)
        response = self.llm.invoke(prompt)
        return response.content


# Example usage
if __name__ == "__main__":
    from rag.config import (
        CHAT_MODEL,
        COLLECTION_NAME,
        PERSIST_DIRECTORY,
        DATA_DIR,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        EMBEDDING_MODEL,
    )
    from rag.logging_config import setup_logging
    from rag.vector_store import VectorStore
    from rag.data_preprocessing import EmbeddingTransformer

    setup_logging()

    # Intialize vectore store
    vector_store = VectorStore(
        persist_dir=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )

    if vector_store.collection.count() == 0:
        from rag.data_ingestion import DataIngestion

        ingestion_handler = DataIngestion(
            data_source=DATA_DIR,
            vector_store=vector_store,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            embedding_model_name=EMBEDDING_MODEL,
        )
        ingestion_handler.ingest_data()

    # Initialize embedding transformer
    embedding_transformer = EmbeddingTransformer(model_name=EMBEDDING_MODEL)

    rag_search = RAGSearch(
        vector_store=vector_store,
        chat_model_name=CHAT_MODEL,
        embedding_transformer=embedding_transformer,
    )

    query = "Explain attention mechanism in transformers"
    answer = rag_search.retrieve_and_generate(query, top_k=3)
    print("Answer:", answer)


    query = "Explain usage of machine learning"
    answer = rag_search.retrieve_and_generate(query, top_k=5)
    print("Answer:", answer)
