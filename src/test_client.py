import requests

def main():
    print("🧠 GigaChat-клиент. Введите 'выход' для завершения.\n")

    while True:
        user_input = input("❓ Введите вопрос: ").strip()
        if user_input.lower() in ["выход", "exit", "quit"]:
            break

        payload = {
            "question": user_input,
            "top_k": 3
        }

        try:
            response = requests.post("http://127.0.0.1:8000/ask", json=payload)
            response.raise_for_status()
            answer = response.json()["answer"]
            print(f"\n🤖 Ответ от GigaChat:\n{answer}\n")

        except requests.exceptions.RequestException as e:
            print(f"\n🚨 Ошибка при обращении к API: {e}")
        except Exception as ex:
            print(f"\n🚨 Неожиданная ошибка: {ex}")

if __name__ == "__main__":
    main()