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
	
	# Remove references sidebar and nav
	for sel in ["nav", ".ltx_page_nav", ".ltx_bibliography", ".ltx_appendix"]:
		for el in soup.select(sel):
			el.decompose()

	sections: List[Dict[str, str]] = []
	
	# First try to get full paper sections from ar5iv documents
	for sec in soup.select(".ltx_document .ltx_section, article .ltx_section, .ltx_section"):
		title_el = sec.find(["h1", "h2", "h3", "h4", "h5", "h6"]) or sec.find(class_="ltx_title")
		title = title_el.get_text(strip=True) if title_el else ""
		
		# Skip empty sections or reference sections
		if not title or title.lower() in ["references", "bibliography", "acknowledgments", "acknowledgements"]:
			continue
			
		section_text = md(str(sec), heading_style="ATX", strip=['script','style'])
		if len(section_text.strip()) > 50:  # Only include substantial sections
			sections.append({
				"title": title,
				"html": str(sec),
				"markdown": section_text
			})
	
	# If we got good sections, return them
	if sections:
		return sections
	
	# Fallback 1: Try to extract key content areas from full document
	main_content = soup.find("article") or soup.find("main") or soup.find(".ltx_document") or soup.find("body")
	if main_content:
		# Extract abstract
		abs_block = main_content.select_one("blockquote.abstract, .ltx_abstract, .abstract")
		if abs_block:
			abstract_text = abs_block.get_text(" ", strip=True)
			if len(abstract_text) > 50:
				sections.append({
					"title": "Abstract",
					"html": str(abs_block),
					"markdown": abstract_text,
				})
		
		# Try to extract main content by paragraphs and headings
		content_elements = main_content.select("p, h1, h2, h3, h4, h5, h6, .ltx_para")
		current_section = "Introduction"
		current_content = []
		
		for elem in content_elements:
			if elem.name in ["h1", "h2", "h3", "h4", "h5", "h6"] or "ltx_title" in elem.get("class", []):
				# Save previous section if it has content
				if current_content:
					section_text = "\n\n".join(current_content)
					if len(section_text.strip()) > 100:  # Minimum length for meaningful content
						sections.append({
							"title": current_section,
							"html": "",
							"markdown": section_text
						})
				# Start new section
				current_section = elem.get_text(strip=True)
				current_content = []
			else:
				# Add paragraph content
				para_text = elem.get_text(" ", strip=True)
				if len(para_text) > 20:  # Skip very short paragraphs
					current_content.append(para_text)
		
		# Don't forget the last section
		if current_content:
			section_text = "\n\n".join(current_content)
			if len(section_text.strip()) > 100:
				sections.append({
					"title": current_section,
					"html": "",
					"markdown": section_text
				})
	
	# Fallback 2: If still no sections, just extract all meaningful text
	if not sections:
		# Final fallback - just get abstract if that's all we can find
		abs_block = soup.select_one("blockquote.abstract")
		if abs_block:
			abstract = abs_block.get_text(" ", strip=True)
			if abstract:
				sections.append({
					"title": "Abstract",
					"html": str(abs_block),
					"markdown": abstract,
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


def _enhanced_abstract_extraction(soup: BeautifulSoup, abstract_text: str) -> List[Dict[str, str]]:
	"""Create multiple structured sections from abstract and page metadata for deeper content."""
	sections = []
	
	# Extract metadata
	title_elem = soup.select_one("h1.title")
	title = title_elem.get_text(strip=True).replace("Title:", "").strip() if title_elem else ""
	
	authors_elem = soup.select_one(".authors")
	authors = authors_elem.get_text(" ", strip=True).replace("Authors:", "").strip() if authors_elem else ""
	
	subjects_elem = soup.select_one(".subjects")
	subjects = subjects_elem.get_text(" ", strip=True) if subjects_elem else ""
	
	comments_elem = soup.select_one(".comments")
	comments = comments_elem.get_text(" ", strip=True) if comments_elem else ""
	
	# Create comprehensive paper overview
	overview_parts = []
	if title:
		overview_parts.append(f"**Title:** {title}")
	if authors:
		overview_parts.append(f"**Authors:** {authors}")
	if subjects:
		overview_parts.append(f"**Subject Areas:** {subjects}")
	if comments:
		overview_parts.append(f"**Additional Information:** {comments}")
	
	if overview_parts:
		sections.append({
			"title": "Paper Overview",
			"html": "",
			"markdown": "\n\n".join(overview_parts)
		})
	
	# Add the abstract
	sections.append({
		"title": "Abstract",
		"html": str(soup.select_one("blockquote.abstract")) if soup.select_one("blockquote.abstract") else "",
		"markdown": abstract_text,
	})
	
	# Parse abstract to create synthetic technical sections
	abstract_lower = abstract_text.lower()
	sentences = [s.strip() for s in abstract_text.split('.') if len(s.strip()) > 20]
	
	# Methodology section
	method_keywords = ["method", "approach", "framework", "algorithm", "model", "technique", "architecture", "design", "propose", "develop", "introduce"]
	method_sentences = [s for s in sentences if any(kw in s.lower() for kw in method_keywords)]
	if method_sentences:
		sections.append({
			"title": "Methodology and Approach",
			"html": "",
			"markdown": f"This paper introduces novel methodological contributions:\n\n" + "\n\n".join(f"• {s.strip()}." for s in method_sentences[:3])
		})
	
	# Technical contributions section
	tech_keywords = ["neural", "learning", "optimization", "training", "loss", "network", "transformer", "attention", "embedding", "feature"]
	tech_sentences = [s for s in sentences if any(kw in s.lower() for kw in tech_keywords)]
	if tech_sentences:
		sections.append({
			"title": "Technical Contributions",
			"html": "",
			"markdown": f"Key technical innovations described:\n\n" + "\n\n".join(f"• {s.strip()}." for s in tech_sentences[:3])
		})
	
	# Results and evaluation section  
	result_keywords = ["result", "evaluation", "experiment", "performance", "accuracy", "improvement", "outperform", "achieve", "demonstrate", "show"]
	result_sentences = [s for s in sentences if any(kw in s.lower() for kw in result_keywords)]
	if result_sentences:
		sections.append({
			"title": "Results and Evaluation",
			"html": "",
			"markdown": f"Experimental results and performance:\n\n" + "\n\n".join(f"• {s.strip()}." for s in result_sentences[:3])
		})
	
	# Application/impact section
	app_keywords = ["application", "real-world", "practical", "implementation", "deployment", "impact", "significance", "important"]
	app_sentences = [s for s in sentences if any(kw in s.lower() for kw in app_keywords)]
	if app_sentences:
		sections.append({
			"title": "Applications and Impact",
			"html": "",
			"markdown": f"Practical applications and significance:\n\n" + "\n\n".join(f"• {s.strip()}." for s in app_sentences[:3])
		})
	
	# If abstract mentions specific domains, create domain-specific insights
	domain_keywords = {
		"computer vision": ["vision", "image", "visual", "detection", "recognition", "segmentation"],
		"natural language processing": ["language", "text", "nlp", "translation", "generation", "understanding"],
		"machine learning": ["learning", "training", "model", "neural", "deep", "classification"],
		"robotics": ["robot", "control", "manipulation", "navigation", "autonomous"],
		"quantum": ["quantum", "qubit", "entanglement", "decoherence"],
		"medical": ["medical", "clinical", "diagnosis", "health", "patient", "disease"],
		"security": ["security", "attack", "defense", "privacy", "cryptographic"]
	}
	
	for domain, keywords in domain_keywords.items():
		if any(kw in abstract_lower for kw in keywords):
			domain_sentences = [s for s in sentences if any(kw in s.lower() for kw in keywords)]
			if domain_sentences:
				sections.append({
					"title": f"{domain.title()} Context",
					"html": "",
					"markdown": f"Domain-specific contributions in {domain}:\n\n" + "\n\n".join(f"• {s.strip()}." for s in domain_sentences[:2])
				})
			break  # Only add one domain section
	
	return sections


def _pdf_fallback_sections(pdf_url: str) -> List[Dict[str, str]]:
	"""Enhanced fallback that creates comprehensive structured content from available sources."""
	try:
		with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent":"Mozilla/5.0"}) as client:
			arxiv_id = pdf_url.split("/")[-1].replace(".pdf", "")
			
			# Try different ar5iv endpoints for full HTML content first
			ar5iv_urls = [
				f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
			]
			
			for url in ar5iv_urls:
				try:
					resp = client.get(url, timeout=30)
					if resp.status_code == 200 and resp.text:
						extracted = _extract_sections(resp.text)
						if extracted and len(extracted) > 2:  # More than just abstract + footer
							logger.info(f"Successfully extracted {len(extracted)} sections from {url}")
							return extracted
				except Exception as e:
					logger.debug(f"Failed to fetch {url}: {e}")
					continue
			
			# Enhanced abstract extraction with synthetic sections
			abs_url = pdf_url.replace("/pdf/", "/abs/").replace(".pdf", "")
			resp = client.get(abs_url)
			soup = BeautifulSoup(resp.text, "html.parser")
			
			abs_div = soup.select_one("blockquote.abstract")
			if abs_div:
				abstract_text = abs_div.get_text(" ", strip=True)
				logger.info(f"Creating enhanced structured sections from abstract for {arxiv_id}")
				return _enhanced_abstract_extraction(soup, abstract_text)
			
			return []
			
	except Exception as e:
		logger.warning(f"Enhanced PDF fallback failed: {e}")
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

			# Use enhanced fallback if we have no sections OR only minimal content (abstract + footer)
			if not sections or (len(sections) <= 2 and any("arxivlabs" in s.get("title", "").lower() for s in sections)):
				logger.info(f"Using enhanced extraction fallback for {arxiv_id}")
				sections = _pdf_fallback_sections(meta.get("pdf_url", f"https://arxiv.org/pdf/{arxiv_id}.pdf"))

			with open(paper_dir / "sections.json", "w", encoding="utf-8") as fsec:
				json.dump(sections, fsec, ensure_ascii=False, indent=2)

			with open(paper_dir / "figures.json", "w", encoding="utf-8") as ffig:
				json.dump(figures, ffig, ensure_ascii=False, indent=2)

	logger.info("Extraction complete")
