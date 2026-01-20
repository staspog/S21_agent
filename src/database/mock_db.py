"""Мок базы данных с заданиями проектов"""
from typing import List, Dict, Optional
from pathlib import Path
import json

from ..utils import get_logger

logger = get_logger(__name__)


class MockProjectDatabase:
    """Мок базы данных для хранения заданий проектов в markdown формате"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Инициализация БД
        
        Args:
            data_dir: Директория с данными проектов (опционально)
        """
        logger.info("Инициализация MockProjectDatabase")
        self.projects: Dict[str, Dict] = {}
        self._load_mock_data()
        logger.info(f"База данных загружена: {len(self.projects)} проектов")
    
    def _load_mock_data(self):
        """Загружает моковые данные проектов"""
        # Примеры проектов с заданиями
        self.projects = {
            "DS_Bootcamp": {
                "name": "DS_Bootcamp",
                "description": "Курс по Data Science и машинному обучению",
                "tasks": """
# Задания DS_Bootcamp

## Модуль 1: Основы Python
- Задание 1.1: Реализовать функцию для обработки данных
- Задание 1.2: Работа с pandas и numpy
- Задание 1.3: Визуализация данных с matplotlib

## Модуль 2: Машинное обучение
- Задание 2.1: Линейная регрессия с нуля
- Задание 2.2: Классификация с помощью sklearn
- Задание 2.3: Кросс-валидация и метрики

## Модуль 3: Глубокое обучение
- Задание 3.1: Нейронная сеть на PyTorch
- Задание 3.2: CNN для классификации изображений
- Задание 3.3: RNN для обработки текста
"""
            },
            "ML_Project_1": {
                "name": "ML_Project_1",
                "description": "Первый проект по машинному обучению",
                "tasks": """
# Задания ML_Project_1

## Этап 1: Подготовка данных
- Загрузка и очистка данных
- Feature engineering
- Разделение на train/test

## Этап 2: Моделирование
- Выбор алгоритма
- Обучение модели
- Подбор гиперпараметров

## Этап 3: Оценка и деплой
- Оценка качества модели
- Подготовка к продакшену
- Документация
"""
            },
            "Проект_на_Си": {
                "name": "Проект_на_Си",
                "description": "Проект по программированию на языке C",
                "tasks": """
# Задания Проект_на_Си

## Лабораторная 1: Основы
- Работа с указателями
- Управление памятью
- Структуры данных

## Лабораторная 2: Алгоритмы
- Реализация сортировок
- Поиск в структурах данных
- Обработка строк

## Лабораторная 3: Системное программирование
- Работа с файлами
- Процессы и потоки
- Сетевое программирование
"""
            },
            "Web_Development": {
                "name": "Web_Development",
                "description": "Курс по веб-разработке",
                "tasks": """
# Задания Web_Development

## Модуль 1: Frontend
- HTML/CSS основы
- JavaScript и DOM
- React компоненты

## Модуль 2: Backend
- REST API на Flask
- Работа с БД
- Аутентификация

## Модуль 3: Деплой
- Docker контейнеризация
- CI/CD pipeline
- Мониторинг
"""
            },
            "DevOps_Practice": {
                "name": "DevOps_Practice",
                "description": "Практика по DevOps инструментам",
                "tasks": """
# Задания DevOps_Practice

## Задание 1: Docker
- Создание Dockerfile
- Docker Compose
- Оптимизация образов

## Задание 2: Kubernetes
- Развертывание приложений
- Service и Ingress
- Мониторинг

## Задание 3: CI/CD
- Настройка GitHub Actions
- Автоматическое тестирование
- Деплой в продакшен
"""
            }
        }
    
    def get_project(self, project_name: str) -> Optional[Dict]:
        """
        Получает информацию о проекте
        
        Args:
            project_name: Название проекта
            
        Returns:
            Словарь с информацией о проекте или None
        """
        logger.debug(f"[DB] Поиск проекта: {project_name}")
        result = self.projects.get(project_name)
        if result:
            logger.debug(f"[DB] Проект найден: {project_name}")
        else:
            logger.warning(f"[DB] Проект не найден: {project_name}")
        return result
    
    def get_all_projects(self) -> List[Dict]:
        """
        Получает список всех проектов с кратким описанием
        
        Returns:
            Список словарей с ключами 'name' и 'description'
        """
        return [
            {"name": proj["name"], "description": proj["description"]}
            for proj in self.projects.values()
        ]
    
    def search_projects(self, query: str) -> List[str]:
        """
        Поиск проектов по запросу (простой поиск по названию и описанию)
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Список названий релевантных проектов
        """
        query_lower = query.lower()
        results = []
        
        for proj_name, proj_data in self.projects.items():
            if (query_lower in proj_name.lower() or 
                query_lower in proj_data["description"].lower()):
                results.append(proj_name)
        
        return results
    
    def get_project_tasks(self, project_name: str) -> Optional[str]:
        """
        Получает задания проекта в markdown формате
        
        Args:
            project_name: Название проекта
            
        Returns:
            Markdown текст с заданиями или None
        """
        logger.debug(f"[DB] Получение заданий для проекта: {project_name}")
        project = self.get_project(project_name)
        if project:
            tasks = project.get("tasks", "")
            if tasks:
                logger.debug(f"[DB] Задания получены для {project_name} (длина: {len(tasks)} символов)")
            else:
                logger.warning(f"[DB] Задания пусты для проекта: {project_name}")
            return tasks
        logger.warning(f"[DB] Проект не найден для получения заданий: {project_name}")
        return None
