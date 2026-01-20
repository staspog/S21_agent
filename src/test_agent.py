"""Тестовый скрипт для проверки работы LangGraph агента"""
import sys
from pathlib import Path

# Добавляем корень проекта в путь для импортов
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import S21Agent

def main():
    print("Инициализация агента...")
    agent = S21Agent()
    print("Агент готов!\n")
    
    # Тестовые запросы
    test_queries = [
        "Какие задания есть в проекте DS_Bootcamp?",
        "Когда следующая встреча с ментором?",
        "Как реализовать нейронную сеть в ML_Project_1?",
        "Где найти информацию о дедлайнах?",
    ]
    
    for query in test_queries:
        print(f"Вопрос: {query}")
        print("-" * 50)
        try:
            result = agent.invoke(query)
            # Новый API возвращает словарь
            if isinstance(result, dict):
                answer = result.get("answer", "")
                if result.get("needs_input"):
                    print(f"Вопрос агента: {result.get('question', '')}")
                else:
                    print(f"Ответ: {answer}")
            else:
                # Старый формат (на случай если что-то пошло не так)
                print(f"Ответ: {result}")
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()
        print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    main()
