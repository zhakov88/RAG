import panel as pn

from rag.config import (
    CHAT_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    PERSIST_DIRECTORY,
)
from rag.data_preprocessing import EmbeddingTransformer
from rag.logging_config import setup_logging
from rag.rag_search import RAGSearch
from rag.vector_store import VectorStore

pn.extension(design="material")
setup_logging()


async def callback(contents):
    response = rag_search.retrieve_and_generate(contents, top_k=5)
    return response


# Intialize vectore store
vector_store = VectorStore(
    persist_dir=PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME,
)

# Ingest data if vector store is empty
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

# Create chat interface
chat_interface = pn.chat.ChatInterface(callback=callback)
pn.Column(
    """
    # RAG Chat Interface
    ### A simple chat interface powered by Retrieval-Augmented Generation (RAG)
    """,
    chat_interface,
).servable()
