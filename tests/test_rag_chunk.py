from src.rc.schemas import RagChunk


def test_rag_chunk_from_record_roundtrip():
    raw = {
        "id": "adm_info_guests_00001",
        "text": "Формы заявки по кампусам\nМосква",
        "metadata": {
            "source": "https://applicant.21-school.ru/guests",
            "url": "https://applicant.21-school.ru/guests",
            "page_title": "Гости",
            "section": "Формы заявки по кампусам",
            "slug": "guests",
            "part": 0,
            "parts": 1,
            "chunk_file": "corpus/adm_info_chunks.jsonl",
        },
    }
    chunk = RagChunk.from_record(raw)
    assert chunk.id == raw["id"]
    assert chunk.metadata.slug == "guests"
    assert chunk.to_record()["metadata"]["slug"] == "guests"
