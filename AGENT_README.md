# LangGraph Agent - Документация

## Описание

Новый LangGraph агент для обработки запросов пользователей с автоматической классификацией на технические и организационные вопросы.

## Архитектура

### Структура проекта

```
src/
├── config/          # Конфигурация приложения
│   ├── __init__.py
│   └── config.py
├── llm/            # LLM клиент
│   ├── __init__.py
│   └── client.py
├── prompts/        # Промпты для агента
│   ├── __init__.py
│   ├── classification.txt
│   ├── project_search.txt
│   ├── project_confirmation.txt
│   ├── technical_answer.txt
│   └── rocketchat_search.txt
├── agents/         # LangGraph агент
│   ├── __init__.py
│   └── agent.py
├── database/       # Мок БД с заданиями проектов
│   ├── __init__.py
│   └── mock_db.py
└── integrations/   # Интеграции (RocketChat)
    ├── __init__.py
    └── rocketchat.py
```

## Граф агента

Агент использует LangGraph с conditional edges для маршрутизации:

1. **Классификация** (`classify`) - определяет тип запроса
2. **Маршрутизация** - в зависимости от типа:
   - **ТЕХНИЧЕСКИЙ** → Поиск проектов → Подтверждение (опционально) → Ответ
   - **ОРГАНИЗАЦИОННЫЙ** → Поиск в RocketChat → Ответ

## Использование

### Через API

```bash
# Запуск API сервера (из корня проекта)
uvicorn src.api_agent:app --host 0.0.0.0 --port 8000 --reload

# Или из директории src
cd src
uvicorn api_agent:app --host 0.0.0.0 --port 8000 --reload
```

Запрос:
```bash
curl -X POST "http://localhost:8000/agent/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "Какие задания есть в проекте DS_Bootcamp?"}'
```

### Программно

```python
from src.agents import S21Agent

agent = S21Agent()
answer = agent.invoke("Какие задания есть в проекте DS_Bootcamp?")
print(answer)
```

### Тестирование

```bash
# Из корня проекта
python -m src.test_agent

# Или
cd src
python test_agent.py
```

## Конфигурация

Настройки в `.env`:

```env
GIGACHAT_API_KEY=your_key_here
ROCKETCHAT_API_KEY=optional
ROCKETCHAT_URL=https://rocketchat.example.com
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
MAX_PROJECT_SUGGESTIONS=5
PROJECT_CONFIRMATION_REQUIRED=true
```

## Компоненты

### LLM Client

Обертка над GigaChat с методами:
- `invoke(prompt, system_prompt)` - простой вызов
- `invoke_structured(prompt, system_prompt, response_format)` - структурированный ответ
- `chain(prompts)` - цепочка вызовов

### Database (Mock)

Мок базы данных с проектами и заданиями:
- `get_project(project_name)` - получить проект
- `get_all_projects()` - список всех проектов
- `get_project_tasks(project_name)` - задания проекта в markdown

### RocketChat Integration

Мок клиента RocketChat:
- `search(query, limit)` - поиск сообщений
- `format_search_results(results)` - форматирование результатов

## Промпты

Все промпты хранятся в `src/prompts/`:
- `classification.txt` - классификация запросов
- `project_search.txt` - поиск релевантных проектов
- `project_confirmation.txt` - подтверждение выбора проектов
- `technical_answer.txt` - формирование технического ответа
- `rocketchat_search.txt` - формирование поискового запроса для RocketChat

## Расширение

### Добавление новых проектов

Отредактируйте `src/database/mock_db.py`, метод `_load_mock_data()`.

### Изменение промптов

Отредактируйте соответствующие файлы в `src/prompts/`.

### Добавление новых типов запросов

1. Обновите `AgentState` с новым типом
2. Добавьте новый узел в граф
3. Обновите conditional edge в `_route_by_type`
