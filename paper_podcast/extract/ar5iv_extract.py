from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from ..config import Settings
from ..utils.logging import get_logger
from ..utils.paths import run_dir, paper_run_dir, ensure_dir


logger = get_logger(__name__)


def _extract_sections(html: str) -> List[Dict[str, str]]:
	soup = BeautifulSoup(html, "html.parser")
	# If this is an arXiv abs page, prefer the abstract block only
	abs_block = soup.select_one("blockquote.abstract")
	if abs_block:
		abstract = abs_block.get_text(" ", strip=True)
		if abstract:
			return [{
				"title": "Abstract",
				"html": str(abs_block),
				"markdown": abstract,
			}]

	# Remove references sidebar and nav
	for sel in ["nav", ".ltx_page_nav"]:
		for el in soup.select(sel):
			el.decompose()

	sections: List[Dict[str, str]] = []
	# ar5iv full documents usually have .ltx_document with nested .ltx_section
	for sec in soup.select(".ltx_document .ltx_section, article .ltx_section"):
		title_el = sec.find(["h1", "h2", "h3", "h4"]) or sec.find(class_="ltx_title")
		title = title_el.get_text(strip=True) if title_el else ""
		sections.append({
			"title": title,
			"html": str(sec),
			"markdown": md(str(sec), heading_style="ATX", strip=['script','style'])
		})
	# Fallback: treat the main article content as one section
	if not sections:
		main = soup.find("article") or soup.find("body")
		if main:
			sections.append({
				"title": "Paper",
				"html": str(main),
				"markdown": md(str(main), heading_style="ATX", strip=['script','style'])
			})
	return sections


def _extract_figures(html: str) -> List[Dict[str, str]]:
	soup = BeautifulSoup(html, "html.parser")
	figs: List[Dict[str, str]] = []
	for figure in soup.select("figure, .ltx_figure"):
		cap = figure.find(class_="ltx_caption") or figure.find("figcaption")
		caption = cap.get_text(" ", strip=True) if cap else ""
		# For MVP, store figure HTML and caption; image assets can be fetched later from PDFs
		figs.append({
			"caption": caption,
			"html": str(figure)
		})
	return figs


def _fetch_ar5iv(arxiv_id: str) -> str:
	# Try ar5iv with redirects, then fallback to labs subdomain
	url = f"https://ar5iv.org/html/{arxiv_id}"
	headers={"User-Agent":"Mozilla/5.0"}
	with httpx.Client(timeout=30, follow_redirects=True, headers=headers) as client:
		resp = client.get(url)
		if resp.status_code == 200 and resp.text:
			return resp.text
		# fallback
		resp2 = client.get(f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}")
		resp2.raise_for_status()
		return resp2.text


def _pdf_fallback_sections(pdf_url: str) -> List[Dict[str, str]]:
	"""Very simple PDF text fallback using arXiv PDF to text via arxiv abs HTML.
	We avoid heavy PDF parsing for MVP; we extract the abstract from the abs page.
	"""
	try:
		with httpx.Client(timeout=30, follow_redirects=True) as client:
			abs_url = pdf_url.replace("/pdf/", "/abs/").replace(".pdf", "")
			resp = client.get(abs_url)
			soup = BeautifulSoup(resp.text, "html.parser")
			abs_div = soup.select_one("blockquote.abstract")
			abstract = abs_div.get_text(" ", strip=True) if abs_div else ""
			sections = []
			if abstract:
				sections.append({
					"title": "Abstract",
					"html": str(abs_div),
					"markdown": abstract,
				})
			return sections
	except Exception as e:
		logger.warning(f"PDF fallback failed: {e}")
		return []


def extract_run(settings: Settings, run_id: str) -> None:
	run_path = run_dir(settings.data_dir, run_id)
	meta_path = run_path / "metadata.jsonl"
	papers_dir = run_path / "papers"
	assets_dir = ensure_dir(settings.assets_dir)

	if not meta_path.exists():
		raise FileNotFoundError(f"Missing metadata: {meta_path}")

	logger.info(f"Extracting ar5iv HTML for run {run_id}")

	with open(meta_path, "r", encoding="utf-8") as f:
		for line in f:
			meta = json.loads(line)
			arxiv_id = meta["arxiv_id"]
			paper_dir = paper_run_dir(settings.data_dir, run_id) / arxiv_id.replace("/", "_")
			ensure_dir(paper_dir)

			sections: List[Dict[str, str]] = []
			figures: List[Dict[str, str]] = []
			try:
				html = _fetch_ar5iv(arxiv_id)
				sections = _extract_sections(html)
				figures = _extract_figures(html)
			except Exception as e:
				logger.warning(f"Failed to fetch ar5iv for {arxiv_id}: {e}")

			if not sections:
				# fallback: abstract from abs page
				sections = _pdf_fallback_sections(meta.get("pdf_url", f"https://arxiv.org/pdf/{arxiv_id}.pdf"))

			with open(paper_dir / "sections.json", "w", encoding="utf-8") as fsec:
				json.dump(sections, fsec, ensure_ascii=False, indent=2)

			with open(paper_dir / "figures.json", "w", encoding="utf-8") as ffig:
				json.dump(figures, ffig, ensure_ascii=False, indent=2)

	logger.info("Extraction complete")
