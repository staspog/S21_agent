# S21 Agent — Чат-бот с GigaChat и FAISS

RAG-система на основе моделей SentenceTransformers и GigaChat, использующий `.md` и `.txt` файлы как контекст. Эмбеддинги индексируются с помощью FAISS. На данный момент знает о проектах на Си, DS_Bootcamp, а также ML_Project_1.

---

## ⚙️ Установка окружения (Python 3.10)

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/yourname/S21_agent.git
cd S21_agent
```

### 2. Создайте виртуальное окружение и активируйте его

- **Для Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\activate
```

- **Для Windows (Git Bash) / WSL / Linux / macOS:**

```bash
python -m venv venv
source venv/Scripts/activate
```

> На macOS и Linux активация виртуального окружения происходит через `source venv/bin/activate`.

### 3. Установите зависимости

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Получение API ключа GigaChat

- Зарегистрируйтесь или войдите на платформу GigaChat (https://gigachat.ai или другой официальный сайт сервиса).
- В личном кабинете найдите раздел управления API ключами.
- Создайте новый API ключ и скопируйте его.

### 5. Настройте переменные окружения

В корне проекта создайте файл `.env` и добавьте в него строку:

```env
GIGACHAT_API_KEY=ваш_ключ_от_gigachat
```

> **Важно:** файл `.env` добавлен в `.gitignore`, чтобы ваши ключи не попали в публичный репозиторий.

---

## 🚀 Запуск приложения

### Консольная версия чат-бота

Запустите чат-бота в интерактивном режиме командой:

```bash
python src/main.py
```

Если в папке `/content/data` нет уже готовых JSON-файлов с чанками (`chunks.json`, `chunks_map.json`, `chunks.index`), бот проиндексирует `.md` и `.txt` файлы из этой папки и начнёт принимать вопросы.
Если JSON-файлы с чанками присутствуют — бот загрузит их и будет использовать для ответов.

---

### Запуск API сервера

Выполните команды:

```bash
cd src
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Это запустит сервер с API, который отвечает на следующие эндпоинты:

- `/docs` — автоматически сгенерированная документация Swagger UI
- `/ask/` — основной эндпоинт для вопросов к чат-боту

---

### Клиент для работы с API

Для взаимодействия с API запустите:

```bash
python src/client.py
```

Он позволит отправлять вопросы и получать ответы через HTTP-запросы к серверу.

---

## 📁 Структура проекта

```
S21_agent/
├── src/
│   ├── main.py           # консольный запуск чат-бота
│   ├── api.py            # API сервер на FastAPI
│   ├── client.py         # клиент для работы с API
│   └── ...
├── requirements.txt      # зависимости
├── .gitignore            # исключения для Git
├── .env.example          # пример переменных окружения
└── content/
    └── data/             # папка для .md и .txt файлов с контекстом
```

---

## 📌 Зависимости

Все зависимости перечислены в `requirements.txt`. Используются:

* `sentence-transformers`
* `faiss-cpu`
* `gigachat`
* `fastapi`
* `uvicorn`
* `markdown`, `uuid`, `dotenv`, `numpy`, `json`

---

## 🔐 Безопасность

* Файл `.env` добавлен в `.gitignore` и не попадает в репозиторий.
* Ключи API загружаются безопасно через библиотеку `python-dotenv`.
* Если не хотите хранить `.env`, можно использовать безопасный ввод ключа в консоли:

```python
from getpass import getpass
api_key = getpass("Введите ключ: ")
```

---

© 2025 — S21 School Assistant
