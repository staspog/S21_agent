import numpy as np

def find_relevant_chunks(query: str, model, index, chunks_map, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    matched_chunks = [chunks_map[str(i)] for i in indices[0]]
    return matched_chunks

def build_prompt(user_query: str, chunks: list[dict]) -> str:
    context_text = "\n\n---\n\n".join(
        f"Источник: {c['source_file']} / {c['section']}\n{c['content']}"
        for c in chunks
    )
    prompt = (
        f"Ты — умный и вежливый помощник. Используй только факты из предоставленного контекста.\n\n"
        f"Контекст:\n{context_text}\n\n"
        f"Вопрос: {user_query}\n\n"
        f"Ответ:"
    )
    return prompt