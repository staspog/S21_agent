# Используем официальный Python образ
FROM python:3.10-slim

# Устанавливаем зависимости ОС (если нужны)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Экспортируем порт 8000
EXPOSE 8000

# Команда для запуска сервера
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]