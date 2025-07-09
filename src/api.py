from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from sentence_transformers import SentenceTransformer
from embeddings import load_index_and_map
from search import find_relevant_chunks, build_prompt
from gigachat_api import ask_gigachat

# Загрузка ключа
load_dotenv()
api_key = os.getenv("GIGACHAT_API_KEY")
if not api_key:
    raise RuntimeError("GIGACHAT_API_KEY не найден в .env")

print("[API] Загружаем модель и FAISS индекс...")
# Загрузка модели и индекса
model = SentenceTransformer("all-MiniLM-L6-v2")
index, chunks_map = load_index_and_map()
print("[API] Готово к приёму запросов.")

# Создание FastAPI приложения
app = FastAPI(title="S21 Agent API")

# Модель запроса
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

# Модель ответа
class QueryResponse(BaseModel):
    answer: str

# Эндпоинт
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        chunks = find_relevant_chunks(request.question, model, index, chunks_map, top_k=request.top_k)
        prompt = build_prompt(request.question, chunks)
        answer = ask_gigachat(api_key, prompt)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))