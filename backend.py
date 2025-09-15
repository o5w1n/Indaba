import os
import pickle
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

embedder = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"

def load_pdf(file_path):
    pdf = PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

pdf_folder = "vault"

all_chunks = []
for filename in os.listdir(pdf_folder):
       if filename.endswith(".pdf"):
           text = load_pdf(os.path.join(pdf_folder, filename))
           chunks = chunk_text(text)
           all_chunks.extend(chunks)

# vectors = embedder.encode(all_chunks)
# vectors = np.array(vectors)

vectors = embedder.encode(all_chunks, batch_size=32, show_progress_bar=True)
vectors = np.array(vectors).astype('float32')
# print(vectors.shape)


dim = vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(vectors)

faiss.write_index(index, INDEX_FILE)
with open(CHUNKS_FILE, "wb") as f:
       pickle.dump(all_chunks, f)
