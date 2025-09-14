from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import httpx
import pandas as pd

from ..config import Settings
from ..utils.logging import get_logger
from ..utils.paths import ensure_dir, run_dir


logger = get_logger(__name__)


ARXIV_API = "https://export.arxiv.org/api/query"


@dataclass
class PaperMeta:
	arxiv_id: str
	title: str
	authors: List[str]
	abstract: str
	categories: List[str]
	published_at: str
	pdf_url: str
	ar5iv_url: str
	primary_category: str
	link: str


def _parse_arxiv_feed(xml_text: str) -> List[PaperMeta]:
	import feedparser

	feed = feedparser.parse(xml_text)
	papers: List[PaperMeta] = []
	for entry in feed.entries:
		arxiv_id = entry.id.split("/abs/")[-1]
		pdf_url = next((l.href for l in entry.links if l.type == "application/pdf"), f"https://arxiv.org/pdf/{arxiv_id}.pdf")
		categories = [t["term"] for t in entry.get("tags", [])]
		primary = entry.get("arxiv_primary_category", {}).get("term", categories[0] if categories else "")
		authors = [a.name for a in entry.get("authors", [])]
		published = entry.get("published", datetime.now(timezone.utc).isoformat())
		papers.append(
			PaperMeta(
				arxiv_id=arxiv_id,
				title=entry.title.strip().replace("\n", " "),
				authors=authors,
				abstract=entry.summary.strip().replace("\n", " "),
				categories=categories,
				published_at=published,
				pdf_url=pdf_url,
				ar5iv_url=f"https://ar5iv.org/html/{arxiv_id}",
				primary_category=primary,
				link=entry.link,
			)
		)
	return papers


async def _fetch_arxiv_async(category: str, max_results: int) -> List[PaperMeta]:
	# arXiv API limits 30000 chars; use a single window for MVP
	params = {
		"search_query": f"cat:{category}",
		"sortBy": "submittedDate",
		"sortOrder": "descending",
		"start": 0,
		"max_results": max_results,
	}
	async with httpx.AsyncClient(timeout=30) as client:
		resp = await client.get(ARXIV_API, params=params)
		resp.raise_for_status()
		return _parse_arxiv_feed(resp.text)


async def _fetch_arxiv_monthly_async(category: str, year: int, month: int) -> List[PaperMeta]:
	"""Fetch all papers for a specific month and category using pagination."""
	from datetime import datetime, timezone
	
	# Create date range for the month
	start_date = datetime(year, month, 1, tzinfo=timezone.utc)
	if month == 12:
		end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
	else:
		end_date = datetime(year, month + 1, 1, tzinfo=timezone.utc)
	
	# Format dates for arXiv API
	start_str = start_date.strftime("%Y%m%d%H%M%S")
	end_str = end_date.strftime("%Y%m%d%H%M%S")
	
	all_papers = []
	start = 0
	batch_size = 2000  # arXiv API max per request
	
	logger.info(f"Fetching papers for {category} from {year}-{month:02d} (start: {start_str}, end: {end_str})")
	
	async with httpx.AsyncClient(timeout=60) as client:
		while True:
			params = {
				"search_query": f"cat:{category} AND submittedDate:[{start_str} TO {end_str}]",
				"sortBy": "submittedDate",
				"sortOrder": "descending",
				"start": start,
				"max_results": batch_size,
			}
			
			logger.info(f"Fetching batch starting at {start}...")
			resp = await client.get(ARXIV_API, params=params)
			resp.raise_for_status()
			
			batch_papers = _parse_arxiv_feed(resp.text)
			if not batch_papers:
				break
				
			all_papers.extend(batch_papers)
			logger.info(f"Fetched {len(batch_papers)} papers in this batch (total: {len(all_papers)})")
			
			# If we got fewer papers than requested, we've reached the end
			if len(batch_papers) < batch_size:
				break
				
			start += batch_size
	
	logger.info(f"Total papers fetched for {year}-{month:02d}: {len(all_papers)}")
	return all_papers


def ingest_arxiv(settings: Settings, run_id: str) -> int:
	"""Fetch latest papers for a category and store metadata for the run."""
	output_dir = run_dir(settings.data_dir, run_id)
	papers_dir = ensure_dir(output_dir / "papers")
	meta_path = output_dir / "metadata.jsonl"
	csv_path = output_dir / "metadata.csv"

	import asyncio

	logger.info(f"Fetching arXiv feed for {settings.field_category} (limit={settings.max_papers_per_run})")
	papers = asyncio.run(_fetch_arxiv_async(settings.field_category, settings.max_papers_per_run))

	# Deduplicate by arxiv_id
	seen = set()
	deduped: List[PaperMeta] = []
	for p in papers:
		if p.arxiv_id in seen:
			continue
		seen.add(p.arxiv_id)
		deduped.append(p)

	records = [asdict(p) for p in deduped]
	# Save JSONL
	with open(meta_path, "w", encoding="utf-8") as f:
		for r in records:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")

	# Save CSV for humans
	pd.DataFrame(records).to_csv(csv_path, index=False)

	# Per-paper directory skeleton
	for r in records:
		paper_dir = papers_dir / r["arxiv_id"].replace("/", "_")
		ensure_dir(paper_dir)
		# Write a tiny index.json for quick lookups
		with open(paper_dir / "index.json", "w", encoding="utf-8") as f:
			json.dump(r, f, ensure_ascii=False, indent=2)

	logger.info(f"Stored {len(records)} papers into {output_dir}")
	return len(records)


def ingest_arxiv_monthly(settings: Settings, year: int, month: int) -> int:
	"""Fetch all papers for a specific month and category."""
	month_id = f"{year}-{month:02d}"
	output_dir = run_dir(settings.data_dir, month_id)
	papers_dir = ensure_dir(output_dir / "papers")
	meta_path = output_dir / "metadata.jsonl"
	csv_path = output_dir / "metadata.csv"

	import asyncio

	logger.info(f"Fetching arXiv feed for {settings.field_category} for {year}-{month:02d}")
	papers = asyncio.run(_fetch_arxiv_monthly_async(settings.field_category, year, month))

	# Deduplicate by arxiv_id
	seen = set()
	deduped: List[PaperMeta] = []
	for p in papers:
		if p.arxiv_id in seen:
			continue
		seen.add(p.arxiv_id)
		deduped.append(p)

	records = [asdict(p) for p in deduped]
	# Save JSONL
	with open(meta_path, "w", encoding="utf-8") as f:
		for r in records:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")

	# Save CSV for humans
	pd.DataFrame(records).to_csv(csv_path, index=False)

	# Per-paper directory skeleton
	for r in records:
		paper_dir = papers_dir / r["arxiv_id"].replace("/", "_")
		ensure_dir(paper_dir)
		# Write a tiny index.json for quick lookups
		with open(paper_dir / "index.json", "w", encoding="utf-8") as f:
			json.dump(r, f, ensure_ascii=False, indent=2)

	logger.info(f"Stored {len(records)} papers for {year}-{month:02d} into {output_dir}")
	return len(records)
