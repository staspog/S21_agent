import os
import glob
import markdown
import re
import uuid
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from gigachat import GigaChat
from gigachat.models import Chat
from dotenv import load_dotenv
import os

load_dotenv()  # загружает переменные из .env
api_key = os.getenv("GIGACHAT_API_KEY")

def strip_markdown(md_text):
    """Удаляет markdown-разметку, но сохраняет заголовки как plain text"""
    # Удаление ссылок [text](url)
    md_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', md_text)
    # Удаление жирного/курсива
    md_text = re.sub(r'\*\*(.*?)\*\*', r'\1', md_text)
    md_text = re.sub(r'\*(.*?)\*', r'\1', md_text)
    # Удаление backticks
    md_text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', md_text)
    # Удаление HTML-тегов
    md_text = re.sub(r'<[^>]+>', '', md_text)
    return md_text

def split_markdown_sections(md_text):
    """Разделяет markdown-файл по заголовкам"""
    sections = []
    current_section = {"title": "Untitled", "content": ""}
    for line in md_text.splitlines():
        if line.strip().startswith("#"):
            # Новый заголовок — начинается новый раздел
            if current_section["content"].strip():
                sections.append(current_section)
            current_section = {"title": line.strip("# ").strip(), "content": ""}
        else:
            current_section["content"] += line + "\n"
    if current_section["content"].strip():
        sections.append(current_section)
    return sections

def load_documents_from_folder(folder_path):
    all_chunks = []

    for filepath in glob.glob(os.path.join(folder_path, "*.md")):
        with open(filepath, encoding="utf-8") as f:
            raw_md = f.read()
        clean_md = strip_markdown(raw_md)
        sections = split_markdown_sections(clean_md)
        for i, sec in enumerate(sections):
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "source_file": os.path.basename(filepath),
                "section": sec["title"],
                "content": sec["content"].strip()
            })

    for filepath in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(filepath, encoding="utf-8") as f:
            raw_text = f.read()
        clean_text = strip_markdown(raw_text)
        all_chunks.append({
            "id": str(uuid.uuid4()),
            "source_file": os.path.basename(filepath),
            "section": "text file",
            "content": clean_text.strip()
        })

    return all_chunks

chunks = load_documents_from_folder("/content/data")

# Фильтрация от слишком маленьких чанков
filtered_chunks = [c for c in chunks if len(c["content"]) > 100]

with open("/content/chunks.json", "w", encoding="utf-8") as f:
    json.dump(filtered_chunks, f, ensure_ascii=False, indent=2)

print(f"Загружено {len(filtered_chunks)} чанков из {len(chunks)}.")

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Загружаем чанки
with open("/content/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

# Загружаем SBERT модель
model = SentenceTransformer("all-MiniLM-L6-v2")

# Получаем список текстов
texts = [chunk["content"] for chunk in chunks]

# Строим эмбеддинги
embeddings = model.encode(texts, show_progress_bar=True)

# Индексация в FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Сохраняем индекс
faiss.write_index(index, "/content/chunks.index")

# Сохраняем сопоставление индексов с ID чанков
id_map = {i: chunk for i, chunk in enumerate(chunks)}

with open("/content/chunks_map.json", "w", encoding="utf-8") as f:
    json.dump(id_map, f, ensure_ascii=False, indent=2)

print("✅ Индексация завершена. Чанков:", len(chunks))

# Загружаем модель
model = SentenceTransformer("all-MiniLM-L6-v2")

# Загружаем индекс и сопоставление чанков
index = faiss.read_index("/content/chunks.index")
with open("/content/chunks_map.json", encoding="utf-8") as f:
    chunks_map = json.load(f)

# Функция поиска ближайших чанков
def find_relevant_chunks(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    matched_chunks = [chunks_map[str(i)] for i in indices[0]]
    return matched_chunks

def build_prompt(user_query, chunks):
    context_text = "\n\n---\n\n".join(
        f"Источник: {c['source_file']} / {c['section']}\n{c['content']}"
        for c in chunks
    )

    prompt = (
        f"Ты — умный и вежливый помощник. Используй только факты из предоставленного контекста. И не говори о сексе и политике\n\n"
        f"Контекст:\n{context_text}\n\n"
        f"Вопрос: {user_query}\n\n"
        f"Ответ:"
    )
    return prompt

def ask_gigachat(prompt_text):
    with GigaChat(credentials=api_key, verify_ssl_certs=False) as giga:
        chat = Chat(messages=[{"role": "user", "content": prompt_text}])
        response = giga.chat(chat)
        return response.choices[0].message.content
    
while True:
    query = input("\n🧑 Введите ваш вопрос (или 'выход'): ")
    if query.lower() in ["выход", "exit", "quit"]:
        break
    chunks = find_relevant_chunks(query)
    prompt = build_prompt(query, chunks)
    answer = ask_gigachat(prompt)
    print(f"\n🤖 GigaChat:\n{answer}")