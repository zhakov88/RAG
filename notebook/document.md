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

### Data Ingestion

```python
### document datastructure
from langchain_core.documents import Document
```

```python
doc = Document(
    page_content="This is the content of the RAG document.",
    metadata={"source": "generated", 
              "author": "LangChain",
              "pages": 10,
              "date_created": "2025-12-27"
              },
)

doc
```

```python
## Create simple txt file
import os
os.makedirs("../data/text_files", exist_ok=True)

sample_text = {
    "../data/text_files/ml_introduction.txt": 
    """Machine Learning (ML) is a subset of artificial intelligence (AI) that enables systems to learn and improve from experience without being explicitly programmed. Below are samples of text generated to explain ML concepts, applications, and technical processes as of late 2025.
1. Basic Definition & Core Concepts
Overview: Machine learning algorithms use statistical techniques to identify patterns in data and make predictions or decisions.
The Four Main Types:
Supervised Learning: Training a model on labeled data (e.g., email spam filtering).
Unsupervised Learning: Finding hidden patterns in unlabeled data.
Semi-Supervised Learning: Using a small amount of labeled data with a large amount of unlabeled data.
Reinforcement Learning: Learning through trial and error to achieve a goal.
2. Technical Workflow Sample
When building an ML-based text classification model in 2025, developers typically follow these steps:
Data Preprocessing: Cleaning text by removing noise and tokenizing sentences into discrete units.
Vectorization: Converting text into numerical vectors using techniques like Word2Vec or sequence vectors to capture semantic relationships.
Training: Feeding these vectors into a model (e.g., an RNN or Transformer) to learn context and predict outputs.
Inference: Using the trained model to process new, unseen data.
3. Sample Text: Modern AI Text Generation
Modern Large Language Models (LLMs) generate text one "token" at a time. These models use autoregressive methods where the previously generated word fragment serves as context for the next prediction.
Example Prompt: "Explain ML in one sentence."
Example Output: "Machine Learning is the science of training computers to recognize complex patterns and make data-driven decisions autonomously.".
4. Real-World Applications
Transportation: ML is used for pricing and predicting arrival times.
Customer Service: NLP models are used to automate responses to inquiries.
Development Tools: ML models are integrated into IDEs to generate code based on natural language descriptions.
For experimentation, libraries like TensorFlow and Hugging Face provide tools and pre-trained models for text generation and classification.
    """
               }

for file_path, content in sample_text.items():
    with open(file_path, "w") as f:
        f.write(content)
```

```python
### Text Loader
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../data/text_files/ml_introduction.txt", encoding="utf-8")
document = loader.load()
document
```

```python
### Directory Loader
from langchain_community.document_loaders import DirectoryLoader

## Load all .txt files from the specified directory
dir_loader = DirectoryLoader("../data/text_files", loader_cls=TextLoader, glob="**/*.txt", show_progress=True)#
documents = dir_loader.load()
documents
```

```python
from langchain_community.document_loaders import PyMuPDFLoader

dir_loader = DirectoryLoader("../data/pdf", loader_cls=PyMuPDFLoader, glob="**/*.pdf", show_progress=True)
pdf_documents = dir_loader.load()
pdf_documents
```
