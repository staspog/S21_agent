import uuid

import requests


def main():
    session_id = str(uuid.uuid4())
    print("GigaChat-клиент. Введите 'выход' для завершения.\n")
    print(f"Сессия (память диалога): {session_id}\n")

    while True:
        user_input = input("Введите вопрос: ").strip()
        if user_input.lower() in ["выход", "exit", "quit"]:
            break

        payload = {
            "question": user_input,
            "top_k": 3,
            "session_id": session_id,
        }

        try:
            response = requests.post("http://127.0.0.1:8000/ask", json=payload)
            response.raise_for_status()
            data = response.json()
            answer = data["answer"]
            if data.get("session_id") and data["session_id"] != session_id:
                session_id = data["session_id"]
                print(f"Сессия обновлена сервером: {session_id}")
            print(f"\nОтвет:\n{answer}\n")

        except requests.exceptions.RequestException as e:
            print(f"\nОшибка при обращении к API: {e}")
        except Exception as ex:
            print(f"\nНеожиданная ошибка: {ex}")


if __name__ == "__main__":
    main()
