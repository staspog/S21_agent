"""Интеграция с RocketChat для поиска организационной информации"""
from typing import List, Dict, Optional
from ..config import ROCKETCHAT_API_KEY, ROCKETCHAT_URL


class RocketChatClient:
    """Мок клиента для работы с RocketChat"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Инициализация клиента RocketChat
        
        Args:
            api_key: API ключ RocketChat
            base_url: Базовый URL RocketChat
        """
        self.api_key = api_key or ROCKETCHAT_API_KEY
        self.base_url = base_url or ROCKETCHAT_URL
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Поиск сообщений в RocketChat
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            
        Returns:
            Список словарей с результатами поиска
        """
        # Мок реализации - в реальности здесь будет API вызов
        # Пример структуры ответа:
        mock_results = [
            {
                "text": f"Найдена информация по запросу '{query}'",
                "channel": "general",
                "author": "mentor",
                "timestamp": "2025-01-15T10:00:00Z",
                "url": f"{self.base_url}/channel/general/message123"
            },
            {
                "text": f"Дополнительная информация: {query}",
                "channel": "announcements",
                "author": "admin",
                "timestamp": "2025-01-14T15:30:00Z",
                "url": f"{self.base_url}/channel/announcements/message456"
            }
        ]
        
        # Фильтруем по релевантности (в реальности это делает API)
        filtered_results = [
            result for result in mock_results 
            if query.lower() in result["text"].lower()
        ]
        
        return filtered_results[:limit]
    
    def format_search_results(self, results: List[Dict]) -> str:
        """
        Форматирует результаты поиска в читаемый текст
        
        Args:
            results: Список результатов поиска
            
        Returns:
            Отформатированный текст с результатами
        """
        if not results:
            return "По вашему запросу ничего не найдено в RocketChat."
        
        formatted = "Найдена следующая информация в RocketChat:\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['text']}\n"
            formatted += f"   Канал: {result['channel']}\n"
            formatted += f"   Автор: {result['author']}\n"
            formatted += f"   Время: {result['timestamp']}\n"
            formatted += f"   Ссылка: {result['url']}\n\n"
        
        return formatted
