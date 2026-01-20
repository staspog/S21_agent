"""Тестовый скрипт для проверки работы LangGraph агента"""
from .agents import S21Agent

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
            answer = agent.invoke(query)
            print(f"Ответ: {answer}")
        except Exception as e:
            print(f"Ошибка: {e}")
        print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    main()
