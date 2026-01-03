---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: RAG
    language: python
    name: python3
---

### RAG Pipeline - Data Ingestion to Vector Store

```python
import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
```

```python
### read all the pdf files inside directory
def process_all_pdf(pdf_directory):
    """process all pdf files in the specified directory and split them into chunks."""
    all_documents = []
    pdf_dir = Path(pdf_directory)

    # Find  all PDF files in the directory
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process.")

    for pdf_file in pdf_files:
        print(f"Processing file: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            # Add source metadata to each document
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages from {pdf_file.name}")

        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")
    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents

all_pdf_documents = process_all_pdf("../data/pdf")
```

```python
all_pdf_documents
```

```python
### Text splitting get into chunks

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks.")
    # Show example of chunk
    if split_docs:
        print("Example chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs
```

```python
chunks = split_documents(all_pdf_documents)
chunks
```

### Embedding and vector store


```python
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
```

```python
class EmbeddingManager:
    """Handles embedding generation using SentenceTransformer."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
    ):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(
                f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
            )
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise e

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
        print(f"Generating embeddings for {len(texts)} texts.")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


## initialize embedding manager
embedding_manager = EmbeddingManager()
embedding_manager
```

```python
class VectorStore:
    def __init__(
        self, collection_name: str, embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db")
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def add_documents(self, documents: List[Dict[str, Any]]):
        texts = [doc["page_content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        embeddings = self.embedding_model.encode(texts).tolist()
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        self.collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts
        )
        print(f"Added {len(texts)} documents to the vector store.")

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_embedding = self.embedding_model.encode([query_text]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        return [
            (doc, score)
            for doc, score in zip(results["documents"][0], results["distances"][0])
        ]
```

```python
class VectorStore:
    """Manages storage of document embeddings using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "../data/vector_store",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF Document Embeddings for RAG"},
            )
            print(f"Vector store initialized with collection: {self.collection_name}")
            print(f"Existing documents in store: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
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
            ids.append(doc_id)  #

            # Prepare metadata, text, and embedding
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["conttent_length"] = len(doc.page_content)
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

vector_store = VectorStore()
vector_store
```

```python
### Convert text to embeddings and add to vector store
texts = [doc.page_content for doc in chunks]

## generate embeddings
embeddings = embedding_manager.generate_embeddings(texts)

## store in vector store
vector_store.add_documents(chunks, embeddings)
```

### Retriever Pipeline


```python
class RAGRetriever:
    """Retrieves relevant documents from the vector store based on a query."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int=5, score_threshold: float=0.0) -> List[Dict[str, Any]]:
        """Retrieve top_k relevant documents for the given query."""
        print(f"Retrieving documents for query: {query}")
        print(f"Using top_k={top_k}, score_threshold={score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Search in vector store
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # Process results
        retrieved_docs = []
        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                similarity_score = 1 - distance  # Convert distance to similarity score
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                        "distance":distance,
                        "rank": i + 1
                    })
            print(f"Retrieved {len(retrieved_docs)} documents after applying score threshold.")
            return retrieved_docs
        
        else:
            print("No documents retrieved from vector store.")
            return []

rag_retriever = RAGRetriever(vector_store, embedding_manager)
rag_retriever
```

```python
rag_retriever.retrieve("What is Deep Learning", top_k=3, score_threshold=0.1)  
```

### Integration Vector Store, RAG Retriever pipeline with LLM output

```python
### Simple RAG pipeline with llama3.2:1b model from ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
```

```python
PROMPT = """                                
You are a helpful assistant. You will be provided with a query and the context.
Your task is to answer the quesion concisely.
                                      
The query is as follows:                    
{input}

The context is as follows:
{context}

Provide a concise and informative response based on the retrieved information.
If you don't know the answer, say "I don't know" (and don't provide a source).

For every piece of information you provide, also provide the source.

Return text as follows:

<Answer to the question>
Source: source_url
"""
```

```python
## initialize ollama llm
llm = ChatOllama(model=os.getenv("CHAT_MODEL", "llama3.2:1b"), temperature=0)  

## Simple RAG pipeline function (rertrieval + generation)
def rag_pipeline(query: str, retriever: RAGRetriever, llm: ChatOllama, top_k: int=5) -> str:
    """Perform RAG pipeline: retrieve relevant documents and generate answer."""
    # Step 1: Retrieve relevant documents
    retrieved_docs = retriever.retrieve(query, top_k=top_k)

    if not retrieved_docs:
        return "No relevant documents found."

    # Step 2: Prepare context for LLM
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])
    prompt = PROMPT.format(input=query, context=context)    

    # Step 3: Generate answer using LLM
    response = llm.invoke(prompt)
    return response.content
```

```python
response = rag_pipeline("How Machine Learning works?", rag_retriever, llm, top_k=3)
```

```python
print(response)
```
