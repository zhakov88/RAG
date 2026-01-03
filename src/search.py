import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from src.vector_store import FaissVectorStore

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:1b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

PROMPT = """                                
You are a helpful assistant. You will be provided with a query and the context.
Your task is to answer the question concisely.
                                      
The query is as follows:                    
{input}

The context is as follows:
{context}

Provide a concise and informative response based on provided context only.
If you don't know the answer, say "I don't know" (and don't provide a source).

For every piece of information you provide, also provide the source from context.

Return text as follows:

<Answer to the question>
Source: doc_id
"""
#Source: source_url

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model=EMBEDDING_MODEL,)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents

            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        self.llm = ChatOllama(model=CHAT_MODEL, temperature=0)  
        print(f"[INFO] LLM initialized: {CHAT_MODEL}")

    def retrieve_and_generate(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = PROMPT.format(input=query, context=context)    
        response = self.llm.invoke(prompt)
        return response.content
