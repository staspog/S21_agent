import numpy as np


def search_chunks_ranked(
    query: str, model, index, chunks_map, top_k: int = 3
) -> tuple[list[dict], list[dict]]:
    """Возвращает чанки в порядке релевантности и метаданные для логов (L2, id, превью)."""
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    chunks: list[dict] = []
    meta: list[dict] = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        i = int(idx)
        c = chunks_map[str(i)]
        chunks.append(c)
        text = (c.get("content") or "").replace("\n", " ").strip()
        preview = text if len(text) <= 140 else text[:140] + "…"
        meta.append(
            {
                "rank": rank,
                "faiss_id": i,
                "l2": float(dist),
                "source": f"{c['source_file']} / {c['section']}",
                "preview": preview,
            }
        )
    return chunks, meta


def find_relevant_chunks(query: str, model, index, chunks_map, top_k=3):
    chunks, _ = search_chunks_ranked(query, model, index, chunks_map, top_k)
    return chunks

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