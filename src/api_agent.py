"""API для LangGraph агента"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import sys
from pathlib import Path

# Добавляем корень проекта в путь для импортов
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import S21Agent

# Загрузка переменных окружения
load_dotenv()

print("[API Agent] Инициализация агента...")
# Инициализация агента
agent = S21Agent()
print("[API Agent] Агент готов к приёму запросов.")

# Создание FastAPI приложения
app = FastAPI(title="S21 Agent API (LangGraph)")

# Модель запроса
class AgentRequest(BaseModel):
    query: str

# Модель ответа
class AgentResponse(BaseModel):
    answer: str
    query_type: str | None = None
    project_names: list[str] | None = None
    search_query: str | None = None

# Эндпоинт для агента
@app.post("/agent/ask", response_model=AgentResponse)
def ask_agent(request: AgentRequest):
    """
    Обрабатывает запрос пользователя через LangGraph агента
    
    Агент автоматически классифицирует запрос на:
    - ТЕХНИЧЕСКИЙ: вопросы по проектам и заданиям
    - ОРГАНИЗАЦИОННЫЙ: вопросы об организации работы
    
    Для технических вопросов агент найдет релевантные проекты и ответит на основе их заданий.
    Для организационных вопросов агент выполнит поиск в RocketChat.
    """
    try:
        answer = agent.invoke(request.query)
        
        # В реальной реализации можно вернуть больше информации о процессе
        return AgentResponse(
            answer=answer,
            query_type=None,  # Можно расширить для возврата типа запроса
            project_names=None,  # Можно расширить для возврата найденных проектов
            search_query=None,  # Можно расширить для возврата поискового запроса
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
def health_check():
    """Проверка работоспособности API"""
    return {"status": "ok", "agent": "ready"}
