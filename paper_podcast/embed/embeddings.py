from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import tiktoken
from openai import OpenAI

from ..config import Settings
from ..utils.logging import get_logger
from ..utils.paths import run_dir


logger = get_logger(__name__)


@dataclass
class Chunk:
	paper_id: str
	section_title: str
	chunk_index: int
	text: str


def _iter_sections(run_path: Path) -> Iterable[tuple[str, list[dict]]]:
	papers_dir = run_path / "papers"
	for paper_dir in papers_dir.iterdir():
		if not paper_dir.is_dir():
			continue
		sec_path = paper_dir / "sections.json"
		if not sec_path.exists():
			continue
		sections = json.loads(sec_path.read_text(encoding="utf-8"))
		yield paper_dir.name, sections


def _chunk_text_by_tokens(text: str, max_tokens: int, enc) -> List[str]:
	# Simple greedy packing by token count
	tokens = enc.encode(text)
	chunks: List[str] = []
	start = 0
	while start < len(tokens):
		end = min(start + max_tokens, len(tokens))
		chunk_text = enc.decode(tokens[start:end])
		chunks.append(chunk_text)
		start = end
	return chunks


def _make_chunks(run_path: Path, target_chunk_tokens: int) -> List[Chunk]:
	enc = tiktoken.get_encoding("cl100k_base")
	chunks: List[Chunk] = []
	for paper_id, sections in _iter_sections(run_path):
		for sec in sections:
			section_title = sec.get("title", "")
			md = sec.get("markdown", "")
			md = md.strip()
			if not md:
				continue
			for idx, text in enumerate(_chunk_text_by_tokens(md, target_chunk_tokens, enc)):
				if not text.strip():
					continue
				chunks.append(Chunk(paper_id=paper_id, section_title=section_title, chunk_index=idx, text=text))
	return chunks


def embed_run(settings: Settings, run_id: str) -> None:
	if not settings.openai_api_key:
		logger.warning("OPENAI_API_KEY not set; skipping embedding.")
		return

	run_path = run_dir(settings.data_dir, run_id)
	vectors_path = run_path / "vectors.parquet"

	chunks = _make_chunks(run_path, settings.target_chunk_tokens)
	if not chunks:
		logger.warning("No chunks found to embed.")
		return

	client = OpenAI(api_key=settings.openai_api_key or None)
	model = settings.openai_embedding_model

	texts = [c.text for c in chunks]
	# batch to respect rate limits
	embeddings: List[List[float]] = []
	batch_size = 128
	for i in range(0, len(texts), batch_size):
		batch = texts[i:i+batch_size]
		resp = client.embeddings.create(model=model, input=batch)
		for d in resp.data:
			embeddings.append(d.embedding)

	df = pd.DataFrame({
		"paper_id": [c.paper_id for c in chunks],
		"section_title": [c.section_title for c in chunks],
		"chunk_index": [c.chunk_index for c in chunks],
		"text": texts,
		"embedding": embeddings,
	})

	df.to_parquet(vectors_path, index=False)
	logger.info(f"Wrote embeddings to {vectors_path}")
