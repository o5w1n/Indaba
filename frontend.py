# import streamlit as st
# import pickle
# import faiss
# from sentence_transformers import SentenceTransformer
# from groq import Groq
# import os
# # from dotenv import load_dotenv

# # load_dotenv()

# INDEX_FILE = "faiss_index.bin"
# CHUNKS_FILE = "chunks.pkl"
# embedder = SentenceTransformer("all-MiniLM-L6-v2",  device="cpu")

# index = faiss.read_index(INDEX_FILE)
# with open(CHUNKS_FILE, "rb") as f:
#     chunks = pickle.load(f)

# client = Groq(api_key=st.secrets["grok"]["api_key"])

# def search_index(query, k=10):
#     q_vec = embedder.encode([query]).astype('float32')
#     D, I = index.search(q_vec, k)
#     return [chunks[i] for i in I[0]]


import streamlit as st
import pickle
import faiss
import os
from groq import Groq

# Constants
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"

# Initialize session state for model
if 'embedder' not in st.session_state:
    try:
        from sentence_transformers import SentenceTransformer
        st.session_state.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception as e:
        st.error(f"Failed to load SentenceTransformer model: {str(e)}")
        st.stop()

# Load index and chunks with error handling
try:
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Required files not found: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"Error loading files: {str(e)}")
    st.stop()

# Initialize Groq client
try:
    client = Groq(api_key=st.secrets["grok"]["api_key"])
except Exception as e:
    st.error("Failed to initialize Groq client. Please check your API key.")
    st.stop()

def search_index(query, k=10):
    try:
        q_vec = st.session_state.embedder.encode([query]).astype('float32')
        D, I = index.search(q_vec, k)
        return [chunks[i] for i in I[0]]
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return []




def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = (
        f"Answer the question based on the context provided. "
        "If the question is not related to the context in any way, do NOT attempt to answer. "
        "Instead, strictly reply: 'My knowledge base does not have information about this.'\n\n"
        f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return response.choices[0].message.content.strip()

st.markdown(
    """
<style>
/* General page styling */
body {
    background-color: #0e0e0e;
    color: #f5f5f5;
    font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* App title */
h1 {
    text-align: center;
    color: #f5f5f5;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: 1px;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}

/* Section headers */
h3 {
    color: #00c6ff;
    border-left: 4px solid #0072ff;
    padding-left: 0.5rem;
    margin-top: 2rem;
}

/* Question input box */
input[type="text"] {
    border: 1px solid #444;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    width: 100%;
    background-color: #1a1a1a;
    color: #f5f5f5;
    font-size: 1rem;
    transition: border 0.3s ease, box-shadow 0.3s ease;
}
input[type="text"]:focus {
    border-color: #00c6ff;
    box-shadow: 0 0 8px rgba(0, 198, 255, 0.4);
    outline: none;
}

/* Buttons */
button[kind="secondary"], button[kind="primary"] {
    border-radius: 20px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white !important;
    border: none;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
button[kind="secondary"]:hover, button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 198, 255, 0.4);
}

/* Retrieved chunks */
.css-1cpxqw2 {  /* Streamlit default class for text blocks */
    background-color: #1a1a1a;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    border: 1px solid #333;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🤖 Indaba")
st.write("Ask questions based on the indexed documents.")

with st.form(key="chat_form", clear_on_submit=True):
    question = st.text_input("Your question:", key="question_input")
    submit_button = st.form_submit_button("Send")

# st.markdown("### 🔍 Retrieved Chunks")
# for i, chunk in enumerate(retrieved, 1):
#     st.write(f"**Chunk {i}:** {chunk[:300]}...")

if submit_button and question:
    retrieved = search_index(question)
    st.markdown("### 🔍 Retrieved Chunks")
    for i, chunk in enumerate(retrieved, 1):
        st.write(f"**Chunk {i}:** {chunk[:300]}...")

    answer = generate_answer(question, retrieved)
    st.markdown("### 🤖 Answer")
    st.write(answer)


if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()