"""LLM клиент для работы с GigaChat"""
from typing import List, Dict, Optional
import time
from gigachat import GigaChat
from gigachat.models import Chat, Messages
from ..config import GIGACHAT_API_KEY, LLM_TEMPERATURE
from ..utils import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Клиент для работы с LLM через GigaChat"""
    
    def __init__(self, api_key: Optional[str] = None, temperature: float = None):
        self.api_key = api_key or GIGACHAT_API_KEY
        self.temperature = temperature or LLM_TEMPERATURE
        if not self.api_key:
            raise ValueError("API ключ не предоставлен")
    
    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Вызов LLM с промптом
        
        Args:
            prompt: Пользовательский промпт
            system_prompt: Системный промпт (опционально)
            
        Returns:
            Ответ от LLM
        """
        start_time = time.time()
        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
        
        logger.debug(f"[LLM] Вызов GigaChat API")
        logger.debug(f"[LLM] Промпт (первые 200 символов): {prompt_preview}")
        if system_prompt:
            logger.debug(f"[LLM] Системный промпт: {system_prompt[:100]}")
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            with GigaChat(credentials=self.api_key, verify_ssl_certs=False) as giga:
                chat = Chat(messages=messages)
                response = giga.chat(chat)
                result = response.choices[0].message.content
                
                elapsed = time.time() - start_time
                result_preview = result[:200] + "..." if len(result) > 200 else result
                logger.info(f"[LLM] Ответ получен (время: {elapsed:.3f}s, длина: {len(result)} символов)")
                logger.debug(f"[LLM] Ответ (первые 200 символов): {result_preview}")
                
                return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[LLM] Ошибка при вызове GigaChat API: {e} (время: {elapsed:.3f}s)")
            raise
    
    def invoke_structured(self, prompt: str, system_prompt: Optional[str] = None, 
                         response_format: Optional[str] = None) -> str:
        """
        Вызов LLM с требованием структурированного ответа
        
        Args:
            prompt: Пользовательский промпт
            system_prompt: Системный промпт
            response_format: Формат ответа (например, "JSON", "одно слово")
            
        Returns:
            Структурированный ответ от LLM
        """
        if response_format:
            full_prompt = f"{prompt}\n\nФормат ответа: {response_format}"
        else:
            full_prompt = prompt
            
        return self.invoke(full_prompt, system_prompt)
    
    def chain(self, prompts: List[Dict[str, str]]) -> str:
        """
        Цепочка вызовов LLM
        
        Args:
            prompts: Список словарей с ключами 'role' и 'content'
            
        Returns:
            Финальный ответ от LLM
        """
        messages = [{"role": msg.get("role", "user"), "content": msg["content"]} 
                   for msg in prompts]
        
        with GigaChat(credentials=self.api_key, verify_ssl_certs=False) as giga:
            chat = Chat(messages=messages)
            response = giga.chat(chat)
            return response.choices[0].message.content
