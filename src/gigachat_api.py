from gigachat import GigaChat
from gigachat.models import Chat

def ask_gigachat(api_key: str, prompt_text: str) -> str:
    with GigaChat(credentials=api_key, verify_ssl_certs=False) as giga:
        chat = Chat(messages=[{"role": "user", "content": prompt_text}])
        response = giga.chat(chat)
        return response.choices[0].message.content