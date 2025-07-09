import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_embeddings(chunks: list[dict], model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [chunk["../content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return model, embeddings

def save_embeddings_and_index(chunks: list[dict], embeddings: np.ndarray, index_path="../content/chunks.index", map_path="../content/chunks_map.json"):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, index_path)

    id_map = {i: chunk for i, chunk in enumerate(chunks)}
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

def load_index_and_map(index_path="../content/chunks.index", map_path="../content/chunks_map.json"):
    index = faiss.read_index(index_path)
    with open(map_path, encoding="utf-8") as f:
        chunks_map = json.load(f)
    return index, chunks_map