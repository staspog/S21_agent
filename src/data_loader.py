import os
import glob
import re
import uuid

def strip_markdown(md_text: str) -> str:
    md_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', md_text)
    md_text = re.sub(r'\*\*(.*?)\*\*', r'\1', md_text)
    md_text = re.sub(r'\*(.*?)\*', r'\1', md_text)
    md_text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', md_text)
    md_text = re.sub(r'<[^>]+>', '', md_text)
    return md_text

def split_markdown_sections(md_text: str) -> list[dict]:
    sections = []
    current_section = {"title": "Untitled", "content": ""}
    for line in md_text.splitlines():
        if line.strip().startswith("#"):
            if current_section["content"].strip():
                sections.append(current_section)
            current_section = {"title": line.strip("# ").strip(), "content": ""}
        else:
            current_section["content"] += line + "\n"
    if current_section["content"].strip():
        sections.append(current_section)
    return sections

def load_documents_from_folder(folder_path: str) -> list[dict]:
    all_chunks = []

    for filepath in glob.glob(os.path.join(folder_path, "*.md")):
        with open(filepath, encoding="utf-8") as f:
            raw_md = f.read()
        clean_md = strip_markdown(raw_md)
        sections = split_markdown_sections(clean_md)
        for sec in sections:
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "source_file": os.path.basename(filepath),
                "section": sec["title"],
                "content": sec["content"].strip()
            })

    for filepath in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(filepath, encoding="utf-8") as f:
            raw_text = f.read()
        clean_text = strip_markdown(raw_text)
        all_chunks.append({
            "id": str(uuid.uuid4()),
            "source_file": os.path.basename(filepath),
            "section": "text file",
            "content": clean_text.strip()
        })

    return all_chunks