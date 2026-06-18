#!/usr/bin/env python3
"""Chunk adm_info markdown pages into RAG-compatible jsonl."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rc.schemas import RagChunk, RagChunkMetadata

DEFAULT_AGENT_ROOT = Path(os.getenv("AGENT_ROOT", str(ROOT))).resolve()
DEFAULT_ADM_INFO_DIR = DEFAULT_AGENT_ROOT / "adm_info"
DEFAULT_OUT = DEFAULT_AGENT_ROOT / "corpus" / "adm_info_chunks.jsonl"

SKIP_FILES = {"README.md"}

ORIGINAL_RE = re.compile(
    r"^>\s*Оригинал:\s*\[[^\]]+\]\(([^)]+)\)\s*$",
    re.MULTILINE,
)
TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
MOJIBAKE_RE = re.compile(r"[ÃÐÑ�]")


log = logging.getLogger("chunk_adm_info")


class AdmPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    rel_path: str
    slug: str
    page_title: str
    page_url: str


class AdmPageRegistry(BaseModel):
    model_config = ConfigDict(frozen=True)

    by_rel: dict[str, AdmPage] = Field(default_factory=dict)

    def lookup_rel(self, rel: str) -> AdmPage | None:
        return self.by_rel.get(rel.replace("\\", "/"))

    def find_page(self, rel: str) -> AdmPage | None:
        rel = rel.replace("\\", "/").strip("/")
        candidates = [rel]
        if rel.endswith(".md"):
            if rel.endswith("/index.md"):
                candidates.append(rel[: -len("/index.md")])
            else:
                candidates.append(rel[: -len(".md")] + "/index.md")
        else:
            candidates.append(f"{rel}/index.md")
            candidates.append(f"{rel}.md")
        seen: set[str] = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            page = self.lookup_rel(cand)
            if page:
                return page
        return None


class SectionBlock(BaseModel):
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


def slug_from_path(path: Path, adm_info_dir: Path) -> str:
    rel = path.relative_to(adm_info_dir)
    if rel.name == "index.md":
        parent = rel.parent
        return parent.as_posix() if parent != Path(".") else "index"
    return rel.with_suffix("").as_posix()


def page_url_from_raw(raw: str, slug: str) -> str:
    """Published applicant URL from ``> Оригинал:``; else empty (internal page only)."""
    url_m = ORIGINAL_RE.search(raw)
    if url_m:
        return url_m.group(1).strip()
    return ""


def page_source_id(page_url: str, slug: str) -> str:
    if page_url.startswith(("http://", "https://")):
        return page_url
    return f"adm_info:{slug}"


def page_title_from_raw(raw: str, slug: str) -> str:
    title_m = TITLE_RE.search(raw)
    return title_m.group(1).strip() if title_m else slug


def collect_markdown_pages(adm_info_dir: Path) -> list[Path]:
    pages = sorted(adm_info_dir.glob("**/*.md"))
    return [p for p in pages if p.name not in SKIP_FILES]


def build_page_registry(pages: list[Path], adm_info_dir: Path) -> AdmPageRegistry:
    by_rel: dict[str, AdmPage] = {}
    for path in pages:
        raw = path.read_text(encoding="utf-8")
        slug = slug_from_path(path, adm_info_dir)
        rel = path.relative_to(adm_info_dir).as_posix()
        by_rel[rel] = AdmPage(
            rel_path=rel,
            slug=slug,
            page_title=page_title_from_raw(raw, slug),
            page_url=page_url_from_raw(raw, slug),
        )
    return AdmPageRegistry(by_rel=by_rel)


def _split_href(href: str) -> tuple[str, str]:
    href = href.strip()
    if "#" not in href:
        return href, ""
    path, anchor = href.split("#", 1)
    anchor = f"#{anchor}" if anchor else ""
    return path.strip(), anchor


def _slug_from_href_path(href_path: str, from_page: Path, adm_info_dir: Path) -> str:
    target = (from_page.parent / href_path).resolve()
    try:
        rel = target.relative_to(adm_info_dir.resolve())
    except ValueError:
        rel = Path(href_path)
    if rel.suffix == ".md":
        return slug_from_path(adm_info_dir / rel, adm_info_dir)
    return slug_from_path(adm_info_dir / rel / "index.md", adm_info_dir)


def resolve_href(
    href: str,
    *,
    from_page: Path,
    adm_info_dir: Path,
    registry: AdmPageRegistry,
) -> tuple[str, str | None]:
    """Resolve markdown href to (url, target_page_title)."""
    href_path, anchor = _split_href(href)
    if not href_path:
        return anchor or href, None

    if href_path.startswith(("http://", "https://", "mailto:", "tel:")):
        return href_path + anchor, None

    if href_path.startswith("#"):
        current = registry.lookup_rel(from_page.relative_to(adm_info_dir).as_posix())
        if current:
            return current.page_url + href_path, current.page_title
        return href, None

    try:
        target = (from_page.parent / href_path).resolve()
        rel = target.relative_to(adm_info_dir.resolve()).as_posix()
    except ValueError:
        log.warning(
            "link outside adm_info: %r in %s",
            href,
            from_page.relative_to(adm_info_dir),
        )
        slug = _slug_from_href_path(href_path, from_page, adm_info_dir)
        return f"{slug}{anchor}", None

    page = registry.find_page(rel)
    if page:
        return page.page_url + anchor, page.page_title

    slug = _slug_from_href_path(href_path, from_page, adm_info_dir)
    log.warning(
        "unresolved adm_info link %r in %s -> %s",
        href,
        from_page.relative_to(adm_info_dir),
        slug,
    )
    return f"{slug}{anchor}", None


def _format_resolved_link(label: str, url: str, target_title: str | None) -> str:
    label = label.strip()
    if not url:
        if target_title and target_title.casefold() != label.casefold():
            return f"{label} (раздел «{target_title}»)"
        return label
    if target_title and target_title.casefold() != label.casefold():
        return f"{label} ({url}, раздел «{target_title}»)"
    return f"{label} ({url})"


def normalize_markdown(
    text: str,
    *,
    from_page: Path,
    adm_info_dir: Path,
    registry: AdmPageRegistry,
) -> str:
    def _link_sub(match: re.Match[str]) -> str:
        label = match.group(1)
        href = match.group(2)
        url, target_title = resolve_href(
            href,
            from_page=from_page,
            adm_info_dir=adm_info_dir,
            registry=registry,
        )
        return _format_resolved_link(label, url, target_title)

    text = LINK_RE.sub(_link_sub, text)
    text = IMAGE_RE.sub(
        lambda m: (m.group(1) or "image").strip() + f" ({m.group(2)})",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_page(
    path: Path,
    adm_info_dir: Path,
    registry: AdmPageRegistry,
) -> tuple[str, str, list[SectionBlock]]:
    raw = path.read_text(encoding="utf-8")
    slug = slug_from_path(path, adm_info_dir)
    page = registry.lookup_rel(path.relative_to(adm_info_dir).as_posix())
    page_title = page.page_title if page else page_title_from_raw(raw, slug)
    page_url = page.page_url if page else page_url_from_raw(raw, slug)

    body = raw
    title_m = TITLE_RE.search(raw)
    if title_m:
        body = body[title_m.end() :]
    body = ORIGINAL_RE.sub("", body, count=1).strip()

    norm_kw = {
        "from_page": path,
        "adm_info_dir": adm_info_dir,
        "registry": registry,
    }

    blocks: list[SectionBlock] = []
    current_section = page_title
    current_lines: list[str] = []

    for line in body.splitlines():
        hm = HEADING_RE.match(line)
        if hm:
            chunk_text = normalize_markdown("\n".join(current_lines), **norm_kw)
            if chunk_text:
                blocks.append(SectionBlock(section=current_section, text=chunk_text))
            current_section = hm.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    chunk_text = normalize_markdown("\n".join(current_lines), **norm_kw)
    if chunk_text:
        blocks.append(SectionBlock(section=current_section, text=chunk_text))

    if not blocks and body.strip():
        blocks.append(
            SectionBlock(
                section=page_title,
                text=normalize_markdown(body, **norm_kw),
            )
        )

    return page_title, page_url, blocks


def page_to_chunks(
    path: Path,
    adm_info_dir: Path,
    registry: AdmPageRegistry,
    *,
    target_words: int,
    overlap_words: int,
) -> list[RagChunk]:
    page_title, page_url, blocks = parse_page(path, adm_info_dir, registry)
    slug = slug_from_path(path, adm_info_dir)
    raw = path.read_text(encoding="utf-8")
    if MOJIBAKE_RE.search(raw):
        log.warning("possible encoding issue: %s", path.relative_to(adm_info_dir))

    chunks: list[RagChunk] = []
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
                RagChunk(
                    id=f"adm_info_{slug.replace('/', '_')}_{part_counter:05d}",
                    text=text.strip(),
                    metadata=RagChunkMetadata(
                        source=page_source_id(page_url, slug),
                        url=page_url or None,
                        page_title=page_title,
                        section=section,
                        slug=slug,
                        part=part_idx,
                        parts=len(parts),
                        chunk_file="corpus/adm_info_chunks.jsonl",
                    ),
                )
            )

    return chunks


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

    pages = collect_markdown_pages(adm_info_dir)
    if not pages:
        log.warning("no markdown pages found under %s", adm_info_dir)

    registry = build_page_registry(pages, adm_info_dir)

    all_chunks: list[RagChunk] = []
    for path in pages:
        page_chunks = page_to_chunks(
            path,
            adm_info_dir,
            registry,
            target_words=args.target_words,
            overlap_words=args.overlap_words,
        )
        log.info("%s -> %s chunks", path.relative_to(adm_info_dir), len(page_chunks))
        all_chunks.extend(page_chunks)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row.to_record(), ensure_ascii=False) + "\n")

    log.info("wrote %s chunks to %s", len(all_chunks), args.out)


if __name__ == "__main__":
    main()
