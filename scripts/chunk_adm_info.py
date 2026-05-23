#!/usr/bin/env python3
"""Chunk adm_info markdown pages into RAG-compatible jsonl."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("chunk_adm_info")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_ROOT = Path(os.getenv("AGENT_ROOT", str(ROOT))).resolve()
DEFAULT_ADM_INFO_DIR = DEFAULT_AGENT_ROOT / "adm_info"
DEFAULT_OUT = DEFAULT_AGENT_ROOT / "corpus" / "adm_info_chunks.jsonl"

SKIP_FILES = {"README.md", "external_links.md"}

ORIGINAL_RE = re.compile(
    r"^>\s*Оригинал:\s*\[[^\]]+\]\(([^)]+)\)\s*$",
    re.MULTILINE,
)
TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
MOJIBAKE_RE = re.compile(r"[ÃÐÑ�]")


@dataclass
class SectionBlock:
    section: str
    text: str


def split_text_words(
    text: str,
    *,
    target_words: int = 240,
    overlap_words: int = 40,
) -> list[str]:
    words = text.split()
    if len(words) <= target_words:
        return [text]

    chunks: list[str] = []
    i = 0
    step = max(1, target_words - overlap_words)
    while i < len(words):
        j = min(len(words), i + target_words)
        chunk = " ".join(words[i:j]).strip()
        if chunk:
            chunks.append(chunk)
        if j >= len(words):
            break
        i += step
    return chunks


def normalize_markdown(text: str) -> str:
    text = LINK_RE.sub(r"\1 (\2)", text)
    text = IMAGE_RE.sub(lambda m: (m.group(1) or "image").strip() + f" ({m.group(2)})", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slug_from_path(path: Path, adm_info_dir: Path) -> str:
    rel = path.relative_to(adm_info_dir)
    parts = list(rel.parts)
    if parts and parts[-1] == "index.md":
        parts = parts[:-1]
    return "/".join(parts) if parts else rel.stem


def parse_page(path: Path, adm_info_dir: Path) -> tuple[str, str, list[SectionBlock]]:
    raw = path.read_text(encoding="utf-8")
    slug = slug_from_path(path, adm_info_dir)

    title_m = TITLE_RE.search(raw)
    page_title = title_m.group(1).strip() if title_m else slug

    url_m = ORIGINAL_RE.search(raw)
    page_url = url_m.group(1).strip() if url_m else f"https://applicant.21-school.ru/{slug}"

    body = raw
    if title_m:
        body = body[title_m.end() :]
    body = ORIGINAL_RE.sub("", body, count=1).strip()

    blocks: list[SectionBlock] = []
    current_section = page_title
    current_lines: list[str] = []

    for line in body.splitlines():
        hm = HEADING_RE.match(line)
        if hm:
            chunk_text = normalize_markdown("\n".join(current_lines))
            if chunk_text:
                blocks.append(SectionBlock(section=current_section, text=chunk_text))
            current_section = hm.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    chunk_text = normalize_markdown("\n".join(current_lines))
    if chunk_text:
        blocks.append(SectionBlock(section=current_section, text=chunk_text))

    if not blocks and body.strip():
        blocks.append(
            SectionBlock(
                section=page_title,
                text=normalize_markdown(body),
            )
        )

    return page_title, page_url, blocks


def page_to_chunks(
    path: Path,
    adm_info_dir: Path,
    *,
    target_words: int,
    overlap_words: int,
) -> list[dict[str, Any]]:
    page_title, page_url, blocks = parse_page(path, adm_info_dir)
    slug = slug_from_path(path, adm_info_dir)
    raw = path.read_text(encoding="utf-8")
    if MOJIBAKE_RE.search(raw):
        log.warning("possible encoding issue: %s", path.relative_to(adm_info_dir))

    chunks: list[dict[str, Any]] = []
    part_counter = 0

    for block in blocks:
        parts = split_text_words(
            block.text,
            target_words=target_words,
            overlap_words=overlap_words,
        )
        for part_idx, part_text in enumerate(parts):
            part_counter += 1
            section = block.section
            text = f"{section}\n{part_text}" if part_text else section
            chunks.append(
                {
                    "id": f"adm_info_{slug.replace('/', '_')}_{part_counter:05d}",
                    "text": text.strip(),
                    "metadata": {
                        "source": page_url,
                        "url": page_url,
                        "page_title": page_title,
                        "section": section,
                        "slug": slug,
                        "part": part_idx,
                        "parts": len(parts),
                        "chunk_file": "corpus/adm_info_chunks.jsonl",
                    },
                }
            )

    return chunks


def collect_pages(adm_info_dir: Path) -> list[Path]:
    pages = sorted(adm_info_dir.glob("**/index.md"))
    return [p for p in pages if p.name not in SKIP_FILES and p.parent != adm_info_dir or p.name == "index.md"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adm-info-dir",
        type=Path,
        default=Path(os.getenv("ADM_INFO_DIR", str(DEFAULT_ADM_INFO_DIR))),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-words", type=int, default=240)
    parser.add_argument("--overlap-words", type=int, default=40)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    adm_info_dir = args.adm_info_dir.resolve()
    if not adm_info_dir.is_dir():
        raise SystemExit(f"adm_info dir not found: {adm_info_dir}")

    pages = [p for p in sorted(adm_info_dir.glob("**/index.md")) if p.name not in SKIP_FILES]
    if adm_info_dir / "index.md" in pages:
        pages.remove(adm_info_dir / "index.md")

    all_chunks: list[dict[str, Any]] = []
    for path in pages:
        page_chunks = page_to_chunks(
            path,
            adm_info_dir,
            target_words=args.target_words,
            overlap_words=args.overlap_words,
        )
        log.info("%s -> %s chunks", path.relative_to(adm_info_dir), len(page_chunks))
        all_chunks.extend(page_chunks)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log.info("wrote %s chunks to %s", len(all_chunks), args.out)


if __name__ == "__main__":
    main()
