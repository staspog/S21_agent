"""Конфигурация приложения"""
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Загружаем переменные окружения
load_dotenv()

# Корень проекта
PROJECT_ROOT = Path(__file__).parent.parent.parent

# API ключи
GIGACHAT_API_KEY: Optional[str] = os.getenv("GIGACHAT_API_KEY")
ROCKETCHAT_API_KEY: Optional[str] = os.getenv("ROCKETCHAT_API_KEY", "")
ROCKETCHAT_URL: Optional[str] = os.getenv("ROCKETCHAT_URL", "https://rocketchat.example.com")

# Настройки LLM
LLM_MODEL: str = os.getenv("LLM_MODEL", "GigaChat")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))

# Настройки агента
MAX_PROJECT_SUGGESTIONS: int = int(os.getenv("MAX_PROJECT_SUGGESTIONS", "5"))
PROJECT_CONFIRMATION_REQUIRED: bool = os.getenv("PROJECT_CONFIRMATION_REQUIRED", "true").lower() == "true"

# Пути к промптам
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts"

# Пути к данным
CONTENT_DIR = PROJECT_ROOT / "content"

# Валидация обязательных параметров
if not GIGACHAT_API_KEY:
    raise RuntimeError("GIGACHAT_API_KEY не найден в .env")
