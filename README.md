# 🧠 RAG Assistant for HR Queries

This Streamlit-based application is a lightweight **Retrieval-Augmented Generation (RAG)** system designed to answer natural language questions about employee data. It embeds structured data into vector form and queries it using cosine similarity before generating a final answer using a local LLM (via Ollama).

## 🔍 What It Does
- Loads employee data from a CSV  
- Creates natural-language fact sentences per employee  
- Embeds those facts using a local transformer model  
- Uses nearest neighbor search to retrieve the top relevant facts  
- Generates an answer using a local LLM (TinyLLaMA via Ollama)  
- Redacts sensitive info like emails and phone numbers from the final output  

## 🛠️ Tech Stack
- **Frontend**: Streamlit  
- **Embedding Model**: SentenceTransformer (MiniLM)  
- **RAG Logic**: Local Nearest Neighbors + Prompting  
- **LLM Inference**: [Ollama](https://ollama.com/) CLI  
- **Language**: Python  

## 📦 Setup Instructions
1. **Install dependencies**  
```bash
pip install -r requirements.txt
```
2. **Run the app**
   ```bash
   streamlit run app.py
```
