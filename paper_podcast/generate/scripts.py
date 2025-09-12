from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..observability.langfuse import get_openai_client

from ..config import Settings
from ..utils.logging import get_logger
from ..utils.paths import run_dir, scripts_dir

# Import observe decorator for Langfuse tracing
try:
    from langfuse import observe
except ImportError:
    # Fallback decorator if Langfuse not available
    def observe():
        def decorator(func):
            return func
        return decorator


logger = get_logger(__name__)


SYSTEM_PROMPT = (
	"You are generating an in-depth, technical dialogue between two expert researchers discussing recent papers. "
	"This is NOT a casual conversation - it's a deep technical dive for an expert audience. "
	"CRITICAL REQUIREMENTS: "
	"1) DEPTH: Discuss technical details, methodologies, mathematical formulations, experimental setups, and algorithmic innovations. "
	"2) TECHNICAL PRECISION: Use proper technical terminology, discuss specific metrics, architectures, and implementation details. "
	"3) CRITICAL ANALYSIS: Analyze strengths, weaknesses, assumptions, limitations, and potential failure modes. "
	"4) COMPARATIVE CONTEXT: Compare approaches with related work, discuss how this advances the field. "
	"5) IMPLEMENTATION INSIGHTS: Discuss practical considerations, computational complexity, scalability issues. "
	"6) CITATIONS: Include precise inline citations [arXiv:ID] after each technical claim. "
	"7) STRUCTURE: For each paper, cover: Problem formulation → Technical approach → Key innovations → Experimental validation → Limitations → Future work. "
	"8) LENGTH: Generate substantial content - aim for detailed technical discussions, not surface-level summaries. "
	"The hosts should demonstrate deep expertise and engage with technical nuances that only domain experts would appreciate."
)


def _format_personas(settings: Settings) -> str:
	h1, h2 = settings.hosts[0], settings.hosts[1]
	return (
		f"Host A: {h1.name} — {h1.style}. Host B: {h2.name} — {h2.style}. "
		"Alternate turns. Avoid filler."
	)


def _build_prompt_for_cluster(cluster: dict, chunk_rows: List[dict], personas: str, minutes_per_section: int) -> List[dict]:
	# Group content by paper for better organization
	papers_content = {}
	for r in chunk_rows:
		paper_id = r['paper_id']
		if paper_id not in papers_content:
			papers_content[paper_id] = []
		papers_content[paper_id].append(f"{r['section_title']}: {r['text']}")
	
	# Build structured context with full paper content
	context_parts = []
	for paper_id, sections in papers_content.items():
		paper_context = f"[arXiv:{paper_id}]\n" + "\n\n".join(sections)
		context_parts.append(paper_context)
	
	context_text = "\n\n---\n\n".join(context_parts)
	
	# Calculate expected word count for deeper content (assuming ~150 words per minute)
	target_words = minutes_per_section * 150
	
	messages = [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user", "content": (
			f"HOST PERSONAS: {personas}\n"
			f"TOPIC CLUSTER: {cluster['label']}\n"
			f"TARGET LENGTH: ~{minutes_per_section} minutes of dialogue (~{target_words} words)\n"
			f"NUMBER OF PAPERS: {len(papers_content)}\n\n"
			"TECHNICAL CONTENT FOR ANALYSIS:\n"
			f"{context_text}\n\n"
			"INSTRUCTIONS:\n"
			"- Generate a substantial technical discussion, not just summaries\n"
			"- Spend significant time on each paper's technical contributions\n"
			"- Discuss mathematical formulations, architectural details, experimental design\n"
			"- Compare and contrast approaches between papers where relevant\n"
			"- Include critical analysis of assumptions, limitations, and potential improvements\n"
			"- Use precise technical vocabulary and discuss implementation considerations\n"
			"- Each paper should get substantial coverage with deep technical insights\n"
			"- Maintain expert-level discourse throughout - assume knowledgeable audience\n\n"
			"Format as a structured dialogue with **Host Name**: before each statement. Include [arXiv:ID] citations after technical claims."
		)},
	]
	return messages


@observe(name="generate_cluster_script")
def _generate_cluster_script(client, cluster: dict, rows: List[dict], personas: str, settings: Settings) -> str:
	"""Generate script for a single cluster with Langfuse tracing."""
	messages = _build_prompt_for_cluster(cluster, rows, personas, settings.minutes_per_section)
	
	resp = client.chat.completions.create(
		model=settings.openai_chat_model,
		messages=messages,
		temperature=0.7,
	)
	
	return resp.choices[0].message.content


@observe(name="generate_podcast_scripts")
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

	# Use Langfuse-instrumented OpenAI client when available
	client = get_openai_client(api_key=settings.openai_api_key or None)
	personas = _format_personas(settings)

	for cluster in clusters:
		paper_ids = set(cluster["paper_ids"]) 
		rows = df[df["paper_id"].isin(paper_ids)].sort_values(["paper_id", "chunk_index"]).to_dict("records")
		if not rows:
			continue
		
		# Generate script for this cluster with tracing
		text = _generate_cluster_script(client, cluster, rows, personas, settings)
		
		out = scripts_path / f"cluster_{cluster['cluster_id']:02d}.md"
		out.write_text(text, encoding="utf-8")
		logger.info(f"Wrote script {out}")

	logger.info("Script generation complete")
