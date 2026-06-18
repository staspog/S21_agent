# S21 Agent — помощник студентов Школы 21 (Сбер)

Чат-бот для ответов по Школе 21. По умолчанию отвечает из **локального RAG-справочника** (FAISS + страницы ADM). Опционально ищет в **Rocket.Chat** (`rocket_search: true`).

## Режимы `/ask`

| `rocket_search` | `deep_search` | Поведение |
|-----------------|---------------|-----------|
| `false` (default) | любой | **RAG-only** — только справочник FAISS; RC не дергается |
| `true` | `false` | **Fast RC** — один подвопрос, BM25-only rerank, урезанные лимиты |
| `true` | `true` | **Deep RC** — decompose, hybrid rerank (BM25 + CrossEncoder), лимиты ×1.2 |

Примечания:

- `deep_search: true` без `rocket_search: true` **игнорируется** (остаётся RAG-only).
- При `rocket_search: true`, если RC недоступен (probe/каталог) — **fallback на RAG** с явной фразой «Rocket.Chat недоступен…».
- Оперативные вопросы про кампус («что сегодня в кампусе») при пустом EVIDENCE — честное «нет дайджеста», без несвязанного RAG.

## Что делает агент

LangGraph-пайплайн:

```
START
  → prepare       (RAG + resolve intent; RC catalog — только если rocket_search=true)
  → [rocket_search?]
      no  → generate (RAG-only)
      yes → probe RC → decompose → fanout → per_subquery (search) → generate
  END
```

Ключевые решения:

- **RAG (default)**: FAISS `IndexFlatIP` + `intfloat/multilingual-e5-small`, top-K из `faiss_store/chunks.jsonl`.
- **Корпус**: peer FAQ, оглавление ADM/FAQ и **полные страницы applicant** из `adm_info/`. Данные **не в public git** — см. ниже.
- **Rocket.Chat (opt-in)**: `chat.search`, RRF, BM25 + CrossEncoder rerank; seed-комнаты — фиксированный список в `src/rc/config.py` (`always_search_room_names`).
- **Память диалога** — `MemorySaver` LangGraph, ключ `session_id` → `thread_id`.
- **Один concurrent `/ask`** — семафор `1`; таймаут графа — `ASK_TIMEOUT_S` (504 при превышении).

### Fast vs Deep (базовые лимиты из env, deep ×1.2)

| Параметр | Fast | Deep (база → ×1.2) |
|----------|------|---------------------|
| Подвопросы | 1 | 2 → 3 |
| Комнат на подвопрос | ≤4 | 6 → 8 |
| Поисковых строк | ≤2 | 2 → 3 |
| Rerank | BM25 only | hybrid |

## Public repo vs private data

Публичный git содержит **только код** (`src/`, `scripts/`, Dockerfile, …).  
Справочник, промпты и индекс RAG в репозиторий **не попадают** — они живут отдельно и на сервере собираются при деплое.

### Что не коммитится

| Путь | Почему |
|------|--------|
| `src/prompts/system.py` | системные промпты (источник — `secrets/materials/agent/`) |
| `faiss_store/` | FAISS-индекс и chunks со текстом справочника |
| `corpus/` | jsonl-чанки FAQ/ADM |
| `adm_info/` | markdown-страницы applicant |
| `docs/*_chunks.jsonl`, `docs.ipynb` | legacy-копии / сырые outputs |
| `.rag_corpus_fingerprint` | отпечаток adm_info/ + corpus/ для deploy (пересборка RAG) |
| `.env` | ключи GigaChat, Rocket.Chat |

Приватные материалы — в **`secrets/materials/agent/`** (монорепо, `.gitignore`):

```
secrets/materials/agent/
  src/prompts/system.py  ← системные промпты
  adm_info/              ← markdown ADM
  corpus/                ← *.jsonl chunks
  faiss_store/           ← опционально локально; на сервере — build при деплое
```

**На сервере:** код rsync из git, materials — из `secrets/materials/`, `faiss_store/` пересобирается там (`build_rag_index.py`) и попадает в Docker-образ. В public remote индекса и корпуса нет.

Локально для разработки:

```bash
cd S21_agent
bash scripts/apply_agent_materials.sh
python scripts/build_rag_index.py --skip-apply
```

Деплой: `scripts/s21_deploy_agent.sh`.

Подробнее про layout materials: [`../secrets/materials/README.md`](../secrets/materials/README.md).

## RAG: пересборка индекса

После правок в `secrets/materials/agent/adm_info/`:

```bash
cd S21_agent
bash scripts/apply_agent_materials.sh
python scripts/build_rag_index.py --skip-apply
```

Или одной командой (apply + chunk + embed):

```bash
python scripts/build_rag_index.py
```

Переменные:

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `AGENT_ROOT` | `S21_agent/` | Корень агента |
| `MATERIALS_AGENT_DIR` | `../secrets/materials/agent` | Приватные materials |
| `ADM_INFO_DIR` | `$AGENT_ROOT/adm_info` | Markdown ADM |
| `RAG_TOPK` | `5` | Сколько чанков достаёт FAISS (в `.env.example` рекомендуется `8`) |
| `RAG_RERANK_TOP` | `0` | CE rerank RAG (`0` = выключен) |
| `RAG_INDEX_PATH` | `faiss_store/index.faiss` | Путь к индексу |
| `RAG_CHUNKS_PATH` | `faiss_store/chunks.jsonl` | Merged chunks |

После пересборки **пересоберите Docker-образ** — `faiss_store/` копируется в образ при build (`Dockerfile`), volume в compose его не монтирует.

Подробнее про режимы и тайминги: [`../S21_AGENT_SPEED_CHANGES.md`](../S21_AGENT_SPEED_CHANGES.md).

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

Обязательные (приложение не стартует без них, даже для RAG-only):

```env
GIGACHAT_API_KEY=...
ROCKETCHAT_BASE_URL=https://rocketchat-student.21-school.ru
ROCKETCHAT_USER=you@student.21-school.ru
ROCKETCHAT_PASSWORD=...
```

См. также `.env.example`.

### API и таймауты

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `ASK_TIMEOUT_S` | `300` | Лимит `/ask` в секундах (504 при превышении) |
| `S21_AGENT_TZ` | `Europe/Moscow` | «Сегодня» в промптах и RC `on:ДД/ММ/ГГГГ` |
| `HOST` / `PORT` / `UVICORN_RELOAD` | `0.0.0.0` / `8000` / `0` | Uvicorn |

### Rocket.Chat — пайплайн

Базовые значения — в `src/rc/config.py`; deep-профиль умножает лимиты на **1.2**.

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `RC_MAX_SUBQUERIES` | `2` | Верхняя граница decompose (deep) |
| `RC_ROOMS_PER_SUBQUERY` | `6` | Комнат на подвопрос (deep); fast ≤4 |
| `RC_QUERIES_PER_SUBQUERY` | `2` | Коротких `searchText` на подвопрос |
| `RC_SEARCH_COUNT` | `20` | Лимит сообщений на один `chat.search` |
| `RC_RERANK_TOP_N` | `8` | Hit-ов после rerank на подвопрос |
| `RC_RERANK_MAX_CANDIDATES` | `15` | Кандидатов в hybrid rerank |
| `RC_RERANK_MODE` | `hybrid` | `hybrid` или `bm25_only` (fast всегда BM25) |
| `RC_BM25_WEIGHT` | `0.4` | Вес BM25 в hybrid |
| `RC_THREAD_EXPAND_TOP` | `2` | Top-hit-ов для раскрытия треда |
| `RC_THREAD_MESSAGES_COUNT` | `30` | Сообщений в треде |
| `RC_HTTP_CONCURRENCY` | `12` | Семафор параллельных REST-вызовов |
| `RC_CATALOG_TTL_S` | `600` | TTL кэша каталога комнат (сек) |
| `RC_SEARCH_TIMEOUT_S` | `20` | Read timeout на `chat.search` |
| `RC_THREAD_TIMEOUT_S` | `15` | Read timeout на треды |
| `RC_TIMEOUT_CONNECT_S` / `RC_TIMEOUT_READ_S` | `10` / `60` | HTTP-таймауты клиента |
| `RC_PROBE_TIMEOUT_S` | `8` | Probe доступности RC (login) |
| `RC_PROBE_CACHE_TTL_S` | `30` | Кэш результата probe |
| `RC_CATALOG_FETCH_TIMEOUT_S` | `25` | Таймаут загрузки каталога в `prepare` |
| `RC_RERANKER_MODEL` | `qilowoq/bge-reranker-v2-m3-en-ru` | CrossEncoder |
| `RC_EVIDENCE_MSG_MAX_CHARS` | `2500` | Обрезка текста сообщения в EVIDENCE (`0` = без лимита) |

Seed-комнаты для поиска задаются в коде (`always_search_room_names` в `config.py`: `general`, `msk_general`, …). Переменная `RC_ALWAYS_SEARCH_ROOMS` в compose зарезервирована, но **пока не подключена** к `load_rc_config`.

## Запуск API

```bash
python main.py
# или
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Docker (nginx edge + app):

```bash
docker compose -f docker-compose.agent.yml up -d --build
```

### Эндпоинты

- `POST /ask` — вопрос пользователя.

  Body:

  ```json
  {
    "question": "...",
    "session_id": "...",
    "include_sources": true,
    "rocket_search": false,
    "deep_search": false
  }
  ```

  Ответ: `{ "answer": "<markdown>", "session_id": "...", "sources": [{ "url": "<permalink>", "label": "<короткая подпись>" }, ...] }`.

- `GET /health` — диагностика:

  ```json
  {
    "status": "ok",
    "rocket_chat_available": true,
    "rocket_chat_error": null,
    "rag_loaded": true,
    "rag_chunks": 286,
    "rag_index_ntotal": 286,
    "rag_index_path_exists": true
  }
  ```

- `GET /docs` — Swagger UI.

За nginx: `GET /healthz` проксируется на `/health` (см. `docker-compose.agent.yml`).

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
│   ├── api.py                # FastAPI, lifespan, /ask, /health
│   ├── graph.py              # LangGraph: prepare → RC branch → generate
│   ├── temporal.py           # «Сегодня», rc_date_anchor on:ДД/ММ/ГГГГ
│   ├── campus_intent.py      # intent «что сегодня в кампусе», digest anchors
│   ├── llm_gate.py           # retry/rate-limit для GigaChat
│   ├── rc/
│   │   ├── config.py         # RocketChatConfig, fast/deep profiles
│   │   ├── client.py         # async REST (login, rooms, chat.search, threads)
│   │   ├── catalog.py        # TTL-кэш каталога + seed_rooms
│   │   ├── availability.py   # probe RC, кэш доступности
│   │   └── schemas.py        # Room, Hit, SearchIntent, FinalAnswer, …
│   ├── pipeline/
│   │   ├── decompose.py      # LLM → SubqueryPlan
│   │   ├── resolve_query.py  # intent + scope из истории
│   │   ├── select_rooms.py   # фиксированный seed-набор комнат
│   │   ├── expand_query.py   # pymorphy3 + LLM → SearchQueries
│   │   ├── search.py         # async chat.search × (rooms × queries)
│   │   ├── fuse.py           # RRF k=60
│   │   ├── rerank.py         # BM25 + CrossEncoder
│   │   ├── rag_faiss.py      # FAISS RAG retrieval
│   │   ├── threads.py        # подгрузка реплаев треда
│   │   └── answer.py         # structured FinalAnswer + блок «Источники»
│   ├── prompts/
│   │   ├── _load.py          # loader (public git)
│   │   └── __init__.py       # API; system.py — private, из secrets
│   └── test_client.py
├── scripts/
│   ├── apply_agent_materials.sh
│   ├── chunk_adm_info.py
│   ├── build_rag_index.py
│   └── s21_deploy_agent.sh
├── corpus/                   # private (gitignored)
├── adm_info/                 # private (gitignored)
├── faiss_store/              # runtime RAG index (gitignored)
├── docker-compose.agent.yml
├── requirements.txt
└── Dockerfile
```

## Безопасность и устойчивость

- Учётные данные RC и GigaChat — только в `.env` (не в git).
- Справочник и RAG-индекс — см. раздел **Public repo vs private data** выше.
- RC: probe при старте и в `prepare`; при недоступности — RAG-fallback, не зависание на catalog.
- Авто-перелогин на 401, обработка 429 с уважением `x-ratelimit-reset`.
- Семафор ограничивает параллельные REST-вызовы к RC.
- LLM structured output: `decompose`, `resolve_query`, `expand_query`, финальный `FinalAnswer`.
- Блок «Источники» с RC permalink собирается **на сервере** по `cited_mids`, не моделью.

© 2026 — S21 Assistant
