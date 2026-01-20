# Оптимизация API для Android приложения

## Реализованные оптимизации

### 1. Асинхронная обработка
- Использование `async/await` для неблокирующих операций
- Позволяет обрабатывать несколько запросов одновременно

### 2. CORS поддержка
- Настроен CORS middleware для работы с Android приложением
- Разрешен доступ с любых источников (в продакшене указать конкретные домены)

### 3. Сессии для интерактивного диалога
- Поддержка многошагового диалога через `session_id`
- Сохранение состояния между запросами
- Автоматическая очистка старых сессий

### 4. Измерение времени обработки
- Возврат `processing_time` для мониторинга производительности
- Помогает оптимизировать UX в Android приложении

## Рекомендации для дальнейшей оптимизации

### 1. Кэширование ответов LLM
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_llm_call(prompt_hash: str):
    # Кэширование частых запросов
    pass
```

### 2. Использование Redis для сессий
```python
import redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Сохранение сессии
redis_client.setex(f"session:{session_id}", 3600, json.dumps(session_state))

# Получение сессии
session_state = json.loads(redis_client.get(f"session:{session_id}"))
```

### 3. Пакетная обработка запросов
- Группировка нескольких запросов в один batch
- Уменьшение количества HTTP запросов

### 4. Streaming ответов
```python
from fastapi.responses import StreamingResponse

@app.post("/agent/ask/stream")
async def ask_agent_stream(request: AgentRequest):
    async def generate():
        # Постепенная отправка ответа по мере генерации
        for chunk in agent.invoke_stream(request.query):
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 5. Connection pooling для БД
- Переиспользование соединений
- Уменьшение задержек при запросах к БД

### 6. Сжатие ответов
```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### 7. Rate limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/agent/ask")
@limiter.limit("10/minute")
async def ask_agent(request: AgentRequest):
    ...
```

## Использование API в Android приложении

### Базовый запрос
```kotlin
data class AgentRequest(
    val query: String,
    val user_response: String? = null,
    val session_id: String? = null
)

data class AgentResponse(
    val answer: String,
    val needs_input: Boolean,
    val question: String?,
    val session_id: String?,
    val processing_time: Double?
)

// Запрос
val request = AgentRequest(query = "первый проект по си")
val response = apiService.askAgent(request)

if (response.needs_input) {
    // Показать вопрос пользователю
    showQuestion(response.question)
    // Сохранить session_id для следующего запроса
    saveSessionId(response.session_id)
} else {
    // Показать ответ
    showAnswer(response.answer)
}
```

### Продолжение диалога
```kotlin
// Ответ на уточняющий вопрос
val request = AgentRequest(
    query = "",
    user_response = "Базовая",
    session_id = savedSessionId
)
val response = apiService.respondToQuestion(request)
```

### Оптимизации на стороне клиента

1. **Кэширование сессий**
   - Сохранять `session_id` локально
   - Не создавать новую сессию для каждого запроса

2. **Retry логика**
   - Автоматический повтор при сетевых ошибках
   - Exponential backoff

3. **Показ индикатора загрузки**
   - Использовать `processing_time` для оценки времени ожидания
   - Показывать прогресс при длительных запросах

4. **Офлайн режим**
   - Кэшировать последние ответы
   - Показывать кэшированные данные при отсутствии сети

## Мониторинг производительности

### Метрики для отслеживания:
- Среднее время обработки запроса
- Количество активных сессий
- Процент запросов, требующих уточняющих вопросов
- Количество ошибок

### Логирование
```python
import logging

logger = logging.getLogger(__name__)

@app.post("/agent/ask")
async def ask_agent(request: AgentRequest):
    logger.info(f"Request: {request.query}, Session: {request.session_id}")
    # ...
    logger.info(f"Response time: {processing_time}s")
```
