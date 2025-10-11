import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path  # 1. Импортируем необходимый класс Path

# 2. Определяем корень проекта. Этот код найдет его вне зависимости от того, где запущен скрипт.
# Path(__file__) -> /app/src/embeddings.py (внутри Docker)
# .parent -> /app/src
# .parent -> /app (это корень нашего проекта в Docker)
PROJECT_ROOT = Path(__file__).parent.parent

# 3. Формируем абсолютные пути к файлам данных, отталкиваясь от корня проекта.
# Теперь эти пути будут правильными всегда.
DEFAULT_INDEX_PATH = PROJECT_ROOT / "content" / "chunks.index"
DEFAULT_MAP_PATH = PROJECT_ROOT / "content" / "chunks_map.json"


def build_embeddings(chunks: list[dict], model_name="all-MiniLM-L6-v2"):
    """Эта функция не меняется, так как не работает с путями."""
    model = SentenceTransformer(model_name)
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return model, embeddings


def save_embeddings_and_index(chunks: list[dict], embeddings: np.ndarray, 
                              index_path=DEFAULT_INDEX_PATH, map_path=DEFAULT_MAP_PATH):
    """
    Сохраняет эмбеддинги и карту чанков, используя динамически определенные пути по умолчанию.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    # Библиотеке faiss лучше передавать путь в виде строки
    faiss.write_index(index, str(index_path))

    id_map = {i: chunk for i, chunk in enumerate(chunks)}
    # Функция open() отлично работает с объектами Path напрямую
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)


def load_index_and_map(index_path=DEFAULT_INDEX_PATH, map_path=DEFAULT_MAP_PATH):
    """
    Загружает индекс и карту чанков, используя динамически определенные пути по умолчанию.
    """
    print(f"[Loader] Загрузка индекса из: {index_path}")
    # Библиотеке faiss лучше передавать путь в виде строки
    index = faiss.read_index(str(index_path))

    print(f"[Loader] Загрузка карты чанков из: {map_path}")
    # Функция open() отлично работает с объектами Path напрямую
    with open(map_path, encoding="utf-8") as f:
        chunks_map = json.load(f)
        
    return index, chunks_map