#!/usr/bin/env python3
"""Merge chunk jsonl files, embed with E5, and rebuild faiss_store/."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("build_rag_index")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AGENT_ROOT = SCRIPT_DIR.parent
DEFAULT_MATERIALS_DIR = DEFAULT_AGENT_ROOT.parent / "secrets" / "materials" / "agent"
MODEL_NAME = "intfloat/multilingual-e5-small"

CHUNK_NAMES = [
    "tonihuy_faq_chunks.jsonl",
    "adm_page_chunks.jsonl",
    "adm_faq_chunks.jsonl",
    "adm_info_chunks.jsonl",
]


def agent_root() -> Path:
    return Path(os.getenv("AGENT_ROOT", str(DEFAULT_AGENT_ROOT))).resolve()


def materials_dir() -> Path:
    return Path(os.getenv("MATERIALS_AGENT_DIR", str(DEFAULT_MATERIALS_DIR))).resolve()


def default_chunk_files(root: Path) -> list[Path]:
    corpus = root / "corpus"
    return [corpus / name for name in CHUNK_NAMES]


def rel_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_chunks(chunk_files: list[Path], *, root: Path) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []
    for path in chunk_files:
        if not path.exists():
            raise FileNotFoundError(f"chunk file missing: {path}")
        rows = read_jsonl(path)
        rel = rel_to_root(path, root)
        for row in rows:
            meta = row.setdefault("metadata", {})
            meta["chunk_file"] = rel
        all_chunks.extend(rows)
        log.info("loaded %s rows from %s", len(rows), rel)
    return all_chunks


def corpus_hash(root: Path) -> str:
    """SHA256 over adm_info/ + corpus/ under AGENT_ROOT (deploy fingerprint)."""
    h = hashlib.sha256()
    for sub in ("adm_info", "corpus"):
        base = root / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix().encode("utf-8")
            h.update(rel)
            h.update(path.read_bytes())
    return h.hexdigest()


def write_corpus_hash(root: Path, digest: str) -> None:
    (root / ".corpus_hash").write_text(digest + "\n", encoding="utf-8")


def apply_materials(root: Path, materials: Path) -> None:
    script = SCRIPT_DIR / "apply_agent_materials.sh"
    env = os.environ.copy()
    env.setdefault("AGENT_ROOT", str(root))
    env.setdefault("MATERIALS_AGENT_DIR", str(materials))
    log.info("running apply_agent_materials.sh")
    subprocess.run(["bash", str(script)], check=True, cwd=root, env=env)


def build_index(
    all_chunks: list[dict[str, Any]],
    *,
    store_dir: Path,
    model_name: str,
    chunk_files: list[Path],
    root: Path,
) -> None:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    store_dir.mkdir(parents=True, exist_ok=True)
    passages = [f"passage: {c['text']}" for c in all_chunks]

    log.info("embedding %s passages with %s", len(passages), model_name)
    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(
        passages,
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    faiss.normalize_L2(emb)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    index_path = store_dir / "index.faiss"
    chunks_path = store_dir / "chunks.jsonl"
    config_path = store_dir / "config.json"

    faiss.write_index(index, str(index_path))
    with chunks_path.open("w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "model": model_name,
        "index_type": "IndexFlatIP",
        "dim": int(dim),
        "normalize_embeddings": True,
        "chunk_files": [rel_to_root(p, root) for p in chunk_files],
        "count": len(all_chunks),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log.info("wrote index=%s chunks=%s config=%s (ntotal=%s)", index_path, chunks_path, config_path, index.ntotal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-root",
        type=Path,
        default=None,
        help="agent root (default: AGENT_ROOT env or S21_agent/)",
    )
    parser.add_argument(
        "--materials-dir",
        type=Path,
        default=None,
        help="secrets/materials/agent (default: MATERIALS_AGENT_DIR env)",
    )
    parser.add_argument(
        "--chunk-file",
        action="append",
        dest="chunk_files",
        type=Path,
        help="chunk jsonl path (repeatable; default: corpus/*.jsonl)",
    )
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--skip-apply",
        action="store_true",
        help="do not run apply_agent_materials.sh",
    )
    parser.add_argument(
        "--skip-chunk-adm-info",
        action="store_true",
        help="do not regenerate corpus/adm_info_chunks.jsonl",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")

    root = (args.agent_root or agent_root()).resolve()
    materials = (args.materials_dir or materials_dir()).resolve()
    store_dir = (args.store_dir or root / "faiss_store").resolve()

    if not args.skip_apply:
        apply_materials(root, materials)

    digest = corpus_hash(root)
    write_corpus_hash(root, digest)
    log.info("corpus hash: %s", digest[:12])

    if not args.skip_chunk_adm_info:
        chunk_script = SCRIPT_DIR / "chunk_adm_info.py"
        env = os.environ.copy()
        env["AGENT_ROOT"] = str(root)
        env["ADM_INFO_DIR"] = str(root / "adm_info")
        log.info("running %s", chunk_script.name)
        subprocess.run([sys.executable, str(chunk_script)], check=True, cwd=root, env=env)

    chunk_files = args.chunk_files or default_chunk_files(root)
    all_chunks = merge_chunks(chunk_files, root=root)
    build_index(
        all_chunks,
        store_dir=store_dir,
        model_name=args.model,
        chunk_files=chunk_files,
        root=root,
    )


if __name__ == "__main__":
    main()
