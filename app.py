import os
import re
import subprocess
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel

OLLAMA_EXE = r"C:\Users\Admin\AppData\Local\Programs\Ollama\ollama.exe"
LOCAL_EMBED_PATH = r"C:\Users\Admin\Internship\Gen AI\models\all-MiniLM-L6-v2"
CSV_PATH = "data.csv"

@st.cache_data
def load_employee_data():
    df = pd.read_csv(CSV_PATH)
    df.drop_duplicates(inplace=True)
    df['name'] = df['name'].str.title().str.strip()
    df['designation'] = df['designation'].str.strip()
    df['manager'] = df['manager'].fillna("Unknown")
    return df

df = load_employee_data()

texts = df.apply(lambda r: f"{r['name']} is {r.designation} reporting to {r.manager}", axis=1).tolist()

@st.cache_resource
def load_embedding_model():
    
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_EMBED_PATH)
    model = AutoModel.from_pretrained(LOCAL_EMBED_PATH)
    model.eval()
    return tokenizer, model

def embed_texts(texts: List[str]) -> np.ndarray:
    tokenizer, model = load_embedding_model()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

embs = embed_texts(texts)
nn = NearestNeighbors(n_neighbors=5, metric='cosine')
nn.fit(embs)

def redact(text: str) -> str:
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "[REDACTED]", text)
    text = re.sub(r"\+?\d[\d\s\-]{7,}\d", "[REDACTED]", text)
    return text

def generate_with_ollama(prompt, model_name = "tinyllama"):
    cmd = [OLLAMA_EXE, "run", model_name, prompt]
    result = subprocess.run(cmd, capture_output=True)
    return result.stdout.decode('utf-8', errors='ignore').strip()

def rag_ask(query: str, top_k: int = 5):
    q_emb = embed_texts([query])
    _, idxs = nn.kneighbors(q_emb, n_neighbors=top_k)
    facts = [texts[i] for i in idxs[0]]
    prompt = (
        "You are a HR assistant. Use the facts below to answer the question.\n\n"
        "Facts:\n" + "\n".join(f"- {f}" for f in facts) +
        f"\n\nQuestion: {query}\nAnswer:"
    )
    answer = generate_with_ollama(prompt)
    return facts, redact(answer)

st.title("RAG Assistant")

st.sidebar.header("Employee Data")
if st.sidebar.checkbox("Show table"):
    st.sidebar.dataframe(df)

query = st.text_input("Enter your question about employees:")
if st.button("Make Query"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("processing..."):
            facts, answer = rag_ask(query)
        st.subheader("Answer")
        st.write(answer)
        # st.subheader("Retrieved Facts")
        for f in facts:
            st.write("- ", f)

