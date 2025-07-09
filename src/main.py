import os
import json
from dotenv import load_dotenv
from data_loader import load_documents_from_folder
from embeddings import build_embeddings, save_embeddings_and_index, load_index_and_map
from search import find_relevant_chunks, build_prompt
from gigachat_api import ask_gigachat
from sentence_transformers import SentenceTransformer

def main():
    load_dotenv()
    api_key = os.getenv("GIGACHAT_API_KEY")

    # Загрузка и подготовка данных (один раз)
    if not os.path.exists("content/chunks.json"):
        chunks = load_documents_from_folder("content/data")
        filtered_chunks = [c for c in chunks if len(c["content"]) > 100]
        with open("content/chunks.json", "w", encoding="utf-8") as f:
            json.dump(filtered_chunks, f, ensure_ascii=False, indent=2)

    with open("content/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    # Построение эмбеддингов и индексирование (один раз)
    if not os.path.exists("content/chunks.index") or not os.path.exists("content/chunks_map.json"):
        model, embeddings = build_embeddings(chunks)
        save_embeddings_and_index(chunks, embeddings)
    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # Загрузка индекса и сопоставления
    index, chunks_map = load_index_and_map()

    while True:
        query = input("\n🧑 Введите ваш вопрос (или 'выход'): ")
        if query.lower() in ["выход", "exit", "quit"]:
            break
        relevant_chunks = find_relevant_chunks(query, model, index, chunks_map)
        prompt = build_prompt(query, relevant_chunks)
        answer = ask_gigachat(api_key, prompt)
        print(f"\n🤖 GigaChat:\n{answer}")

if __name__ == "__main__":
    main()