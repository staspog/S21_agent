# S21 Agent — поиск по Rocket.Chat для Школы 21 (Сбер)

Чат-бот для быстрого поиска информации по Школе 21. Никакой статической
RAG-индексации больше нет — агент идёт напрямую в Rocket.Chat и собирает ответ
из живых сообщений.

## Что делает агент

LangGraph-пайплайн на каждый запрос пользователя:

```
START
  → decompose_query        (LLM → 1..5 независимых подвопросов)
  → fetch_room_catalog     (TTL-кэш каталога RC)
  → fanout (Send per subquery)
      ↳ select_rooms       (LLM по name + topic выбирает 3..5 комнат)
      ↳ expand_query       (pymorphy3 + LLM → 2..4 поисковых строк 1–3 слова)
      ↳ chat.search × N    (rooms × queries параллельно, bounded)
      ↳ rrf_fuse           (k=60, дедуп по mid)
      ↳ hybrid_rerank      (BM25 + cross-encoder bge-reranker-v2-m3-en-ru)
      ↳ thread_expand      (top-3: chat.getThreadMessages)
  → generate (GigaChat)    (markdown-ответ + блок «Источники» с permalink)
  END
```

Ключевые решения:

- **Декомпозиция и выбор комнат через structured output GigaChat** (`with_structured_output(SubqueryPlan|RoomSelection|SearchQueries)`).
- **`chat.search` Rocket.Chat — substring search**, поэтому пользовательский запрос лемматизируется и сводится к 1–3 словам.
- **RRF (Reciprocal Rank Fusion, k=60)** склеивает rooms × queries в один список.
- **Гибридный rerank на CPU**: `BM25 + CrossEncoder` (`qilowoq/bge-reranker-v2-m3-en-ru`), формула `final = w·BM25 + (1-w)·CE`.
- **Память диалога** — `MemorySaver` LangGraph, ключ `session_id` → `thread_id`.
- **Цитаты обязательны**: в ответе инлайн `[rc:<mid>]`, в конце блок «## Источники» — список permalink-ов на сообщения Rocket.Chat.

## Установка (Python 3.10+)

```bash
git clone <repo>
cd S21_agent
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

При первом запуске будет загружена модель `qilowoq/bge-reranker-v2-m3-en-ru` (~720 МБ),
кэш сохранится в стандартный HF cache (`~/.cache/huggingface`).

## Переменные окружения (`.env`)

Обязательные:

```env
GIGACHAT_API_KEY=...
ROCKETCHAT_BASE_URL=https://rocketchat-student.21-school.ru
ROCKETCHAT_USER=you@student.21-school.ru
ROCKETCHAT_PASSWORD=...
```

Опциональные (со значениями по умолчанию в `src/rc/config.py`):

| Переменная                    | По умолчанию                          | Описание                                                  |
|-------------------------------|---------------------------------------|------------------------------------------------------------|
| `RC_MAX_SUBQUERIES`           | `5`                                   | Верхняя граница декомпозиции вопроса                       |
| `RC_ROOMS_PER_SUBQUERY`       | `4`                                   | Сколько комнат выбираем на подвопрос                       |
| `RC_QUERIES_PER_SUBQUERY`     | `3`                                   | Сколько коротких searchText генерим на подвопрос           |
| `RC_SEARCH_COUNT`             | `30`                                  | Лимит сообщений на один `chat.search`                      |
| `RC_HTTP_CONCURRENCY`         | `8`                                   | Семафор параллельных REST-вызовов                          |
| `RC_CATALOG_TTL_S`            | `600`                                 | TTL кэша каталога комнат (сек)                             |
| `RC_RERANK_TOP_N`             | `8`                                   | Сколько hit-ов оставляем после реранка на подвопрос        |
| `RC_BM25_WEIGHT`              | `0.4`                                 | Вес BM25 в гибридной формуле                               |
| `RC_THREAD_EXPAND_TOP`        | `3`                                   | Сколько top-hit-ов разворачиваем тредом                    |
| `RC_THREAD_MESSAGES_COUNT`    | `80`                                  | Лимит сообщений в треде                                    |
| `RC_RERANKER_MODEL`           | `qilowoq/bge-reranker-v2-m3-en-ru`    | Hugging Face id cross-encoder реранкера                    |
| `RC_ALWAYS_SEARCH_ROOMS`      | `msk_adm`                             | Имена seed-комнат через запятую — ищем в них всегда       |
| `RC_TIMEOUT_CONNECT_S` / `RC_TIMEOUT_READ_S` | `5` / `30`              | HTTP таймауты                                              |
| `HOST` / `PORT` / `UVICORN_RELOAD` | `0.0.0.0` / `8000` / `1`          | Параметры запуска uvicorn                                  |

## Запуск API

```bash
python main.py
# или
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Эндпоинты:

- `POST /ask` — вопрос пользователя.
  Body: `{ "question": "...", "session_id": "...", "include_sources": true }`.
  Ответ: `{ "answer": "<markdown>", "session_id": "...", "sources": ["<permalink>", ...] }`.
- `GET /health` — `{ "status": "ok" }`.
- `GET /docs` — Swagger UI.

## Smoke-клиент

```bash
python src/test_client.py
```

Клиент держит один `session_id` на сессию, чтобы сохранялась память диалога.

## Структура

```
S21_agent/
├── main.py
├── src/
│   ├── api.py                # FastAPI + lifecycle (warmup модели)
│   ├── graph.py              # LangGraph: Send-fanout, reducer add
│   ├── rc/
│   │   ├── config.py         # RocketChatConfig из env
│   │   ├── client.py         # async REST-клиент (login, list_rooms, chat.search, threads)
│   │   ├── catalog.py        # TTL-кэш каталога
│   │   └── schemas.py        # Pydantic v2: Room, Hit, SubqueryPlan, ...
│   ├── pipeline/
│   │   ├── decompose.py      # LLM → SubqueryPlan
│   │   ├── select_rooms.py   # LLM по каталогу → RoomSelection
│   │   ├── expand_query.py   # pymorphy3 + LLM → SearchQueries
│   │   ├── search.py         # async chat.search × (rooms × queries)
│   │   ├── fuse.py           # RRF k=60
│   │   ├── rerank.py         # BM25 + CrossEncoder
│   │   ├── threads.py        # подгрузка реплаев треда
│   │   └── answer.py         # промпт + GigaChat без structured output
│   └── test_client.py
├── requirements.txt
└── Dockerfile
```

## Безопасность и устойчивость

- Учётные данные RC хранятся только в `.env`.
- Авто-перелогин на 401, обработка 429 с уважением `x-ratelimit-reset`.
- Семафор ограничивает параллельные REST-вызовы.
- Все LLM-вызовы внутри графа (`decompose`, `select_rooms`, `expand_query`) — через `with_structured_output(PydanticModel)`. Финальный ответ — обычный markdown для рендера на фронте.

© 2026 — S21 School Assistant
