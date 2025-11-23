import streamlit as st
import os
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY in Streamlit Secrets!")

# GEMINI client setup
genai.configure(api_key=GEMINI_API_KEY)

MODEL_ID = "gemini-2.5-flash"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4

# PDF Text Extraction
def pdf_to_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for pg in reader.pages:
        try:
            text = pg.extract_text() or ""
        except:
            text = ""
        pages.append(text)
    return "\n".join(pages)

# Chunking
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Embedding model
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

embedder = load_embedder()

# Build Chroma vector store
def build_vector_store(chunks):
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    return Chroma.from_texts(chunks, embedder, persist_directory="chroma_db")

# Query PDF with Gemini
def query_pdf_with_gemini(pdf_text, user_query):
    chunks = chunk_text(pdf_text)
    if not chunks:
        raise ValueError("No text found in PDF.")

    store = build_vector_store(chunks)
    docs = store.similarity_search(user_query, k=TOP_K)
    retrieved_chunks = [d.page_content for d in docs]

    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{user_query}

Answer concisely:
"""

    response = genai.GenerativeModel(MODEL_ID).generate_content(prompt)
    answer = response.text

    return answer, retrieved_chunks

# Streamlit UI
st.set_page_config(page_title="PDF Chat", layout="wide")
st.title("📘 PDF Chat with Gemini")
st.write("Upload a PDF and ask questions!")

uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])
user_query = st.text_input("Ask a question about the PDF:")

if uploaded_pdf:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    pdf_text = pdf_to_text("uploaded.pdf")

    if user_query:
        with st.spinner("Thinking…"):
            try:
                answer, retrieved_chunks = query_pdf_with_gemini(pdf_text, user_query)
                st.subheader("Answer:")
                st.write(answer)

                with st.expander("Context Used"):
                    for i, ch in enumerate(retrieved_chunks, 1):
                        st.markdown(f"**Chunk {i}:** {ch[:500]}…")
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Upload a PDF to start.")
