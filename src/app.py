from src.search import RAGSearch

if __name__ == "__main__":
    rag = RAGSearch(persist_dir="faiss_store")
    query = "What is machine learning?"
    answer = rag.retrieve_and_generate(query, top_k=5)
    print(f"Answer:\n{answer}")
