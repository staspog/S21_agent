"""Промпты для агента"""
from pathlib import Path
from ..config import PROMPTS_DIR


def load_prompt(filename: str) -> str:
    """Загружает промпт из файла"""
    prompt_path = PROMPTS_DIR / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Промпт не найден: {prompt_path}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def format_prompt(template: str, **kwargs) -> str:
    """Форматирует промпт с параметрами"""
    return template.format(**kwargs)
