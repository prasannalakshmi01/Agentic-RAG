# 🚀 Agentic RAG System using LangGraph

## 📌 Overview
This project implements an **Agentic Retrieval-Augmented Generation (RAG)** system using LangGraph.

Unlike traditional RAG pipelines, this system introduces **decision-making, self-correction, and evaluation mechanisms** to improve answer quality.

---

## 🔍 Features
- LLM-based query routing (vector database vs web search)
- Document relevance filtering using LLM-as-a-judge
- Query rewriting for improved retrieval
- Grounded answer generation with source citations
- Hallucination detection and response validation
- Iterative retry mechanism

---

## 🧠 Agentic Workflow
1. User submits a query  
2. LLM routes the query (vectorstore or web search)  
3. Relevant documents are retrieved  
4. Documents are filtered using LLM grading  
5. Query is rewritten if needed  
6. Answer is generated using context  
7. Answer is validated (grounded + useful)  
8. System retries if response is not satisfactory  

---

## 📊 Architecture

![Agentic RAG Architecture](RAG_Architectcure.png)

The system follows an **agentic loop**:
- Retrieve → Evaluate → Improve → Generate → Validate → Retry

---

## ⚙️ Tech Stack
- LangGraph  
- LangChain  
- FAISS (Vector Database)  
- Google Gemini (LLM)  
- HuggingFace Embeddings  
- Streamlit  

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Agentic-RAG.git
cd Agentic-RAG
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Add API Keys

Create a .env file in the root folder:

GOOGLE_API_KEY=your_api_key
TAVILY_API_KEY=your_api_key
HUGGINGFACE_HUB_TOKEN=your_api_key_here

### 4. Add Input Document

👉 Place your own PDF file in the project directory.

Example:

your_document.pdf

👉 Update the file path inside app.py if needed:

SOURCE_DOCUMENT_PATH = "your_document.pdf"

### 5. Run the application
streamlit run app.py
