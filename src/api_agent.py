"""API для LangGraph агента"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import sys
from pathlib import Path
from typing import Optional
import time

# Добавляем корень проекта в путь для импортов
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import S21Agent
from src.utils import get_logger

logger = get_logger(__name__)

# Загрузка переменных окружения
load_dotenv()

logger.info("Инициализация API агента...")
# Инициализация агента
agent = S21Agent()
logger.info("API агент готов к приёму запросов.")

# Создание FastAPI приложения
app = FastAPI(title="S21 Agent API (LangGraph)")

# CORS для Android приложения
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модель запроса
class AgentRequest(BaseModel):
    query: str
    user_response: Optional[str] = None  # Ответ на уточняющий вопрос
    session_id: Optional[str] = None  # ID сессии для продолжения диалога

# Модель ответа
class AgentResponse(BaseModel):
    answer: str
    needs_input: bool = False  # Нужен ли ответ от пользователя
    question: Optional[str] = None  # Вопрос для пользователя
    session_id: Optional[str] = None  # ID сессии для следующего запроса
    query_type: Optional[str] = None
    project_names: Optional[list[str]] = None
    processing_time: Optional[float] = None  # Время обработки в секундах

# Простое хранилище сессий (в продакшене использовать Redis или БД)
sessions = {}

# Эндпоинт для агента
@app.post("/agent/ask", response_model=AgentResponse)
async def ask_agent(request: AgentRequest, http_request: Request):
    """
    Обрабатывает запрос пользователя через LangGraph агента
    
    Поддерживает интерактивный диалог с уточняющими вопросами.
    Используйте session_id для продолжения диалога.
    
    Агент автоматически классифицирует запрос на:
    - ТЕХНИЧЕСКИЙ: вопросы по проектам и заданиям
    - ОРГАНИЗАЦИОННЫЙ: вопросы об организации работы
    
    Для технических вопросов агент может задать до 3 уточняющих вопросов
    для точного поиска проекта.
    """
    start_time = time.time()
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    logger.info(f"[API] ===== Новый запрос /agent/ask =====")
    logger.info(f"[API] IP: {client_ip}")
    logger.info(f"[API] Запрос: {request.query[:200]}")
    logger.info(f"[API] Session ID: {request.session_id}")
    logger.info(f"[API] User response: {request.user_response}")
    
    try:
        # Получаем состояние сессии, если есть
        session_state = None
        if request.session_id and request.session_id in sessions:
            logger.debug(f"[API] Восстановление сессии: {request.session_id}")
            session_state = sessions[request.session_id]
        else:
            logger.debug(f"[API] Новая сессия")
        
        # Обрабатываем запрос
        result = agent.invoke(
            user_query=request.query,
            user_response=request.user_response,
            session_state=session_state,
            session_id=request.session_id
        )
        
        # Сохраняем состояние сессии
        session_id = request.session_id or result.get("session_state", {}).get("session_id") or f"session_{int(time.time() * 1000)}"
        sessions[session_id] = result.get("session_state", {})
        
        logger.info(f"[API] Сессия сохранена: {session_id}")
        logger.debug(f"[API] Активных сессий: {len(sessions)}")
        
        # Очищаем старые сессии (простая реализация, в продакшене использовать TTL)
        if len(sessions) > 1000:
            logger.warning(f"[API] Превышен лимит сессий ({len(sessions)}), очистка старых")
            # Удаляем самые старые сессии
            oldest_keys = list(sessions.keys())[:100]
            for key in oldest_keys:
                del sessions[key]
            logger.info(f"[API] Удалено {len(oldest_keys)} старых сессий")
        
        processing_time = time.time() - start_time
        
        response = AgentResponse(
            answer=result.get("answer", ""),
            needs_input=result.get("needs_input", False),
            question=result.get("question"),
            session_id=session_id,
            query_type=result.get("session_state", {}).get("query_type"),
            project_names=result.get("session_state", {}).get("project_names"),
            processing_time=round(processing_time, 3),
        )
        
        logger.info(f"[API] ===== Запрос обработан успешно =====")
        logger.info(f"[API] Session ID: {session_id}")
        logger.info(f"[API] Needs input: {response.needs_input}")
        logger.info(f"[API] Processing time: {response.processing_time}s")
        
        return response
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[API] Ошибка обработки запроса: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Ошибка обработки запроса: {str(e)}"
        )

# Эндпоинт для ответа на уточняющий вопрос
@app.post("/agent/respond", response_model=AgentResponse)
async def respond_to_question(request: AgentRequest, http_request: Request):
    """
    Отправка ответа на уточняющий вопрос агента
    
    Используйте этот эндпоинт, когда needs_input=true в предыдущем ответе.
    """
    start_time = time.time()
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    logger.info(f"[API] ===== Ответ на вопрос /agent/respond =====")
    logger.info(f"[API] IP: {client_ip}")
    logger.info(f"[API] Session ID: {request.session_id}")
    logger.info(f"[API] User response: {request.user_response}")
    
    if not request.session_id:
        logger.error(f"[API] Ошибка: session_id не предоставлен")
        raise HTTPException(
            status_code=400, 
            detail="session_id обязателен для ответа на вопрос"
        )
    
    if not request.user_response:
        logger.error(f"[API] Ошибка: user_response не предоставлен")
        raise HTTPException(
            status_code=400, 
            detail="user_response обязателен для ответа на вопрос"
        )
    
    try:
        # Получаем состояние сессии
        if request.session_id not in sessions:
            logger.error(f"[API] Сессия не найдена: {request.session_id}")
            raise HTTPException(
                status_code=404, 
                detail="Сессия не найдена"
            )
        
        logger.debug(f"[API] Восстановление сессии: {request.session_id}")
        session_state = sessions[request.session_id]
        
        # Обрабатываем ответ
        result = agent.invoke(
            user_query="",  # Не используется при ответе на вопрос
            user_response=request.user_response,
            session_state=session_state,
            session_id=request.session_id
        )
        
        # Обновляем состояние сессии
        sessions[request.session_id] = result.get("session_state", {})
        logger.debug(f"[API] Сессия обновлена: {request.session_id}")
        
        processing_time = time.time() - start_time
        
        response = AgentResponse(
            answer=result.get("answer", ""),
            needs_input=result.get("needs_input", False),
            question=result.get("question"),
            session_id=request.session_id,
            query_type=result.get("session_state", {}).get("query_type"),
            project_names=result.get("session_state", {}).get("project_names"),
            processing_time=round(processing_time, 3),
        )
        
        logger.info(f"[API] ===== Ответ обработан успешно =====")
        logger.info(f"[API] Session ID: {request.session_id}")
        logger.info(f"[API] Needs input: {response.needs_input}")
        logger.info(f"[API] Processing time: {response.processing_time}s")
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[API] Ошибка обработки ответа: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Ошибка обработки ответа: {str(e)}"
        )

# Health check
@app.get("/health")
def health_check():
    """Проверка работоспособности API"""
    logger.debug("[API] Health check запрос")
    return {
        "status": "ok", 
        "agent": "ready",
        "active_sessions": len(sessions)
    }

# Эндпоинт для очистки сессии
@app.delete("/agent/session/{session_id}")
def clear_session(session_id: str):
    """Очистка сессии"""
    logger.info(f"[API] Запрос на удаление сессии: {session_id}")
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"[API] Сессия удалена: {session_id}")
        return {"status": "deleted", "session_id": session_id}
    else:
        logger.warning(f"[API] Попытка удалить несуществующую сессию: {session_id}")
        raise HTTPException(status_code=404, detail="Сессия не найдена")
