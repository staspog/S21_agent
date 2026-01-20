"""Конфигурация логирования для проекта"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Создаем директорию для логов
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Формат логов
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Формат для файлов
FILE_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | [%(filename)s:%(lineno)d] | %(message)s"


def setup_logger(name: str, level: int = logging.INFO, log_to_file: bool = True) -> logging.Logger:
    """
    Настройка логгера для модуля
    
    Args:
        name: Имя логгера (обычно __name__)
        level: Уровень логирования
        log_to_file: Логировать ли в файл
        
    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Удаляем существующие обработчики
    logger.handlers.clear()
    
    # Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Файловый обработчик
    if log_to_file:
        # Файл для всех логов
        file_handler = logging.FileHandler(
            LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(FILE_LOG_FORMAT, DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Отдельный файл для ошибок
        error_handler = logging.FileHandler(
            LOGS_DIR / f"errors_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Получить логгер для модуля"""
    return logging.getLogger(name)


# Настройка корневого логгера
root_logger = setup_logger("s21_agent", logging.INFO)
