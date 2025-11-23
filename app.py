import streamlit as st
import os
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------- Load Environment Variables --------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY environment variable with your Gemini API key.")

# -------- Configure Gemini --------
genai.configure(api_key=GEMINI_API_KEY)

# -------- Configurations --------
MODEL_ID = "gemini-2.5-flash"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4

# -------- PDF Text Extraction --------
def pdf_to_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for pg in reader.pages:
        try:
            text = pg.extract_text() or ""
        except Exception:
            text = ""
        pages.append(text)
    return "\n".join(pages)

# -------- Text Chunking --------
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -------- Initialize Embeddings --------
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

embedder = load_embedder()

# -------- Build FAISS Vector Store --------
def build_vector_store(chunks):
    return FAISS.from_texts(chunks, embedder)

# -------- Query Pipeline --------
def query_pdf_with_gemini(pdf_text: str, user_query: str):
    chunks = chunk_text(pdf_text)
    if not chunks:
        raise ValueError("No text found in PDF.")

    store = build_vector_store(chunks)
    docs = store.similarity_search(user_query, k=TOP_K)
    retrieved_chunks = [d.page_content for d in docs]

    context = "\n\n---\n\n".join(retrieved_chunks)

    system_prompt = (
        "You are an assistant that answers the user's question strictly using the provided PDF context. "
        "If the answer is not present, say 'Not found in the PDF' and summarize relevant parts."
    )

    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser Question: {user_query}\n\nAnswer:"

    response = genai.GenerativeModel(MODEL_ID).generate_content(prompt)

    answer = response.text
    return answer, retrieved_chunks

# -------- Streamlit UI --------
st.set_page_config(page_title="PDF Chat App", layout="wide")
st.title("PDF Chat App")
st.markdown("Upload a PDF, ask a question, and get context-aware answers using Gemini 2.5 Flash + FAISS + Sentence Transformers.")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
user_query = st.text_input("Ask a question about this PDF:")

if uploaded_pdf is not None:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    pdf_text = pdf_to_text("uploaded.pdf")

    if user_query:
        with st.spinner("Searching PDF..."):
            try:
                answer, retrieved_chunks = query_pdf_with_gemini(pdf_text, user_query)
                st.subheader("Answer:")
                st.write(answer)

                with st.expander("View extracted context chunks"):
                    for i, ch in enumerate(retrieved_chunks, start=1):
                        st.markdown(f"**Chunk {i}:** {ch[:500]}...")

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload a PDF to begin.")
