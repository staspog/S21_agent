
# S21 Agent — Чат-бот с GigaChat и FAISS

Интеллектуальный ассистент на основе моделей SentenceTransformers и GigaChat, использующий `.md` и `.txt` файлы как контекст. Эмбеддинги индексируются с помощью FAISS.

---

## ⚙️ Установка окружения (Python 3.10)

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/yourname/S21_agent.git
cd S21_agent
```

### 2. Создайте виртуальное окружение

```bash
python -m venv venv
source venv/Scripts/activate  # для Git Bash / WSL / Linux
```

### 3. Установите зависимости

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Настройте переменные окружения

Создайте файл `.env` в корне проекта:

```env
GIGACHAT_API_KEY=ваш_ключ_от_gigachat
```

---

## 🚀 Запуск

Подготовьте данные:

* Поместите `.md` и `.txt` файлы в папку `/content/data` (создайте вручную, если нужно).

Запустите чат-бота:

```bash
python main.py
```

Бот проиндексирует документы и начнёт принимать вопросы в интерактивном режиме.

---

## 📁 Структура проекта

```
S21_agent/
├── main.py               # основной скрипт
├── requirements.txt      # зависимости
├── .gitignore            # исключения для Git
├── .env.example          # пример переменных окружения
└── /content
    ├── chunks.json       # извлечённые секции
    ├── chunks_map.json   # сопоставление чанков и ID
    └── chunks.index      # FAISS-индекс
```

---

## 📌 Зависимости

Все зависимости перечислены в `requirements.txt`. Используются:

* `sentence-transformers`
* `faiss-cpu`
* `gigachat`
* `markdown`, `uuid`, `dotenv`, `numpy`, `json`

---

## 🔐 Безопасность

* `.env` добавлен в `.gitignore` и не попадает в репозиторий.
* Ключи загружаются безопасно через `dotenv`.

---

## 🧠 Подсказка

Если не хотите хранить `.env`, можно использовать безопасный ввод:

```python
from getpass import getpass
api_key = getpass("Введите ключ: ")
```

---

© 2025 — S21 School Assistant
