# S21 Agent — Чат-бот с GigaChat и FAISS

RAG-система на основе SentenceTransformers и GigaChat с оркестрацией через LangGraph и памятью диалога в рамках сессии (`session_id` / checkpoint по `thread_id`). Контекст берётся из проиндексированных `.md` и `.txt` файлов; поиск по FAISS.

---

## Установка окружения (Python 3.10+)

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/yourname/S21_agent.git
cd S21_agent
```

### 2. Создайте виртуальное окружение и активируйте его

- **Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\activate
```

- **Windows (Git Bash) / WSL / Linux / macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Установите зависимости

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Получение API ключа GigaChat

- Зарегистрируйтесь или войдите на платформу GigaChat (https://gigachat.ai или другой официальный сайт сервиса).
- В личном кабинете создайте API ключ.

### 5. Переменные окружения

В корне проекта создайте файл `.env`:

```env
GIGACHAT_API_KEY=ваш_ключ_от_gigachat
```

> Файл `.env` в `.gitignore`, ключи не попадают в репозиторий.

---

## Запуск

### API-сервер (FastAPI)

Из корня репозитория (обязательно — иначе ломаются относительные импорты в `src.api`):

```bash
python main.py
```

Переменные окружения (необязательно): `HOST`, `PORT`, `UVICORN_RELOAD` (`0` / `false` — без auto-reload).

Альтернатива:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Не запускайте `uvicorn api:app` из каталога `src` — модуль окажется вне пакета и импорт `from .agent_graph` упадёт.

Эндпоинты:

- `/docs` — Swagger UI
- `POST /ask` — вопрос к боту; тело JSON: `question`, опционально `top_k`, опционально `session_id` (если не указан, сервер создаст новый и вернёт в ответе)

### Клиент для проверки API

С запущенным сервером:

```bash
python src/test_client.py
```

Клиент создаёт один `session_id` на время работы, чтобы сохранялся контекст диалога.

---

## Структура проекта

```
S21_agent/
├── main.py               # точка входа: python main.py
├── src/
│   ├── api.py              # FastAPI, вызов LangGraph
│   ├── agent_graph.py      # граф RAG (retrieve → generate)
│   ├── embeddings.py       # FAISS и карта чанков
│   ├── search.py           # поиск чанков по запросу
│   ├── data_loader.py      # загрузка документов для индексации
│   └── test_client.py      # простой HTTP-клиент
├── requirements.txt
├── Dockerfile
└── content/                # индекс и данные (chunks.index, chunks_map.json, data/ …)
```

---

## Зависимости

Основное перечислено в `requirements.txt`, в том числе:

- `sentence-transformers`, `faiss-cpu`
- `fastapi`, `uvicorn`
- `langgraph`, `langchain-gigachat`, `langchain-core`
- `python-dotenv`

---

## Безопасность

- `.env` не коммитится; ключи подхватываются через `python-dotenv`.

---

© 2026 — S21 School Assistant
