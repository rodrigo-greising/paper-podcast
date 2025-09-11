from __future__ import annotations

import json
from pathlib import Path
from typing import List

from openai import OpenAI

from ..config import Settings
from ..utils.logging import get_logger
from ..utils.paths import run_dir, scripts_dir


logger = get_logger(__name__)


SYSTEM_PROMPT = (
	"You are generating a long-form, citation-grounded dialogue between two hosts about research papers. "
	"Follow these rules: 1) Only make claims supported by provided excerpts; 2) Include inline citations like [arXiv:ID] after claims; "
	"3) Use a clear structure: Intro, then sections per topic, each with background context, key methods, results, limitations, and future directions; "
	"4) Distinct voices: Host A and Host B speak in consistent personas; 5) Keep segments concise but substantive."
)


def _format_personas(settings: Settings) -> str:
	h1, h2 = settings.hosts[0], settings.hosts[1]
	return (
		f"Host A: {h1.name} — {h1.style}. Host B: {h2.name} — {h2.style}. "
		"Alternate turns. Avoid filler."
	)


def _build_prompt_for_cluster(cluster: dict, chunk_rows: List[dict], personas: str, minutes_per_section: int) -> List[dict]:
	context = []
	for r in chunk_rows:
		context.append(f"[arXiv:{r['paper_id']}] {r['section_title']}: {r['text']}")
	context_text = "\n\n".join(context[:60])  # cap context per cluster
	messages = [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user", "content": (
			f"Personas: {personas}\n"
			f"Topic label: {cluster['label']}\n"
			f"Target length: ~{minutes_per_section} minutes of dialogue.\n"
			"Context excerpts (with arXiv ids):\n" + context_text + "\n\n"
			"Write a scripted dialogue labeled with speaker names, with inline [arXiv:ID] citations."
		)},
	]
	return messages


def generate_run(settings: Settings, run_id: str) -> None:
	if not settings.openai_api_key:
		logger.warning("OPENAI_API_KEY not set; skipping generation.")
		return

	run_path = run_dir(settings.data_dir, run_id)
	vectors_path = run_path / "vectors.parquet"
	clusters_path = run_path / "clusters.json"
	scripts_path = scripts_dir(settings.data_dir, run_id)

	if not vectors_path.exists() or not clusters_path.exists():
		logger.warning("Missing vectors or clusters. Run embed and cluster first.")
		return

	import pandas as pd

	df = pd.read_parquet(vectors_path)
	with open(clusters_path, "r", encoding="utf-8") as f:
		clusters = json.load(f)

	if df.empty or not clusters:
		logger.warning("No data to generate scripts.")
		return

	client = OpenAI(api_key=settings.openai_api_key or None)
	personas = _format_personas(settings)

	for cluster in clusters:
		paper_ids = set(cluster["paper_ids"]) 
		rows = df[df["paper_id"].isin(paper_ids)].sort_values(["paper_id", "chunk_index"]).to_dict("records")
		if not rows:
			continue
		messages = _build_prompt_for_cluster(cluster, rows, personas, settings.minutes_per_section)
		resp = client.chat.completions.create(
			model=settings.openai_chat_model,
			messages=messages,
			temperature=0.7,
		)
		text = resp.choices[0].message.content
		out = scripts_path / f"cluster_{cluster['cluster_id']:02d}.md"
		out.write_text(text, encoding="utf-8")
		logger.info(f"Wrote script {out}")

	logger.info("Script generation complete")
