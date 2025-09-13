from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from ..observability.langfuse import get_openai_client
from ..config import Settings
from ..utils.logging import get_logger
from ..utils.paths import run_dir, scripts_dir, paper_run_dir, ensure_dir
from ..extract.ar5iv_extract import _fetch_ar5iv, _extract_sections, _extract_figures, _enhanced_abstract_extraction
from ..ingest.arxiv_ingest import PaperMeta

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


DEEP_DIVE_SYSTEM_PROMPT = (
    "You are generating an in-depth, comprehensive technical analysis of a SINGLE research paper. "
    "This is a deep dive podcast episode focusing entirely on one paper, designed for expert researchers and practitioners. "
    "CRITICAL REQUIREMENTS: "
    "1) COMPREHENSIVE COVERAGE: Discuss every major section of the paper in detail - introduction, methodology, experiments, results, limitations. "
    "2) TECHNICAL DEPTH: Explain mathematical formulations, architectural details, algorithmic innovations, implementation specifics. "
    "3) CRITICAL ANALYSIS: Thoroughly analyze strengths, weaknesses, assumptions, experimental design choices, statistical significance. "
    "4) CONTEXTUAL UNDERSTANDING: Place this work in the broader research landscape, compare with related work, discuss implications. "
    "5) PRACTICAL INSIGHTS: Discuss reproducibility, computational requirements, potential applications, real-world deployment considerations. "
    "6) METHODOLOGICAL SCRUTINY: Examine experimental setup, baselines, evaluation metrics, dataset choices, potential biases. "
    "7) FUTURE DIRECTIONS: Speculate on follow-up work, limitations that need addressing, potential improvements. "
    "8) CITATIONS: Include the paper's arXiv ID [arXiv:ID] when referencing specific claims or results. "
    "9) LENGTH: This is a SINGLE PAPER deep dive - generate substantial content (aim for 15-20 minutes of dialogue). "
    "10) STRUCTURE: Cover the paper systematically - motivation/background → technical approach → experiments → results → analysis → implications. "
    "The hosts should demonstrate deep domain expertise and engage with nuances that only expert researchers would appreciate. "
    "This is not a summary - it's a comprehensive technical analysis and discussion."
)


def _format_personas(settings: Settings) -> str:
    h1, h2 = settings.hosts[0], settings.hosts[1]
    return (
        f"Host A: {h1.name} — {h1.style}. Host B: {h2.name} — {h2.style}. "
        "Alternate turns naturally. Avoid filler phrases."
    )


def _find_paper_in_runs(settings: Settings, arxiv_id: str) -> Optional[tuple[str, Path, Dict]]:
    """Find a paper across all runs. Returns (run_id, paper_dir, metadata)."""
    data_dir = Path(settings.data_dir)
    
    # First check if arxiv_id looks like a run_id format (YYYY-MM-DD)
    if len(arxiv_id.split('-')) == 3 and all(part.isdigit() for part in arxiv_id.split('-')):
        logger.info(f"Interpreting {arxiv_id} as run_id, not arxiv_id")
        return None
    
    # Search through all runs
    runs_dir = data_dir / "runs"
    if not runs_dir.exists():
        return None
        
    for run_dir_path in runs_dir.iterdir():
        if not run_dir_path.is_dir():
            continue
            
        run_id = run_dir_path.name
        papers_dir = run_dir_path / "papers"
        
        if not papers_dir.exists():
            continue
            
        # Try to find paper by exact arxiv_id match
        paper_dir = papers_dir / arxiv_id.replace("/", "_")
        if paper_dir.exists():
            index_file = paper_dir / "index.json" 
            if index_file.exists():
                with open(index_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                return run_id, paper_dir, metadata
                
        # If not found by directory name, search through all papers in this run
        for paper_subdir in papers_dir.iterdir():
            if not paper_subdir.is_dir():
                continue
                
            index_file = paper_subdir / "index.json"
            if index_file.exists():
                try:
                    with open(index_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    
                    # Check if this is the paper we're looking for
                    if metadata.get("arxiv_id") == arxiv_id or metadata.get("arxiv_id", "").replace("v1", "") == arxiv_id.replace("v1", ""):
                        return run_id, paper_subdir, metadata
                        
                except (json.JSONDecodeError, KeyError):
                    continue
                    
    return None


def _retrieve_full_paper_content(arxiv_id: str, paper_dir: Path, settings: Settings) -> List[Dict[str, str]]:
    """Retrieve the most complete version of paper content."""
    
    # First try to load existing sections
    sections_file = paper_dir / "sections.json"
    if sections_file.exists():
        with open(sections_file, "r", encoding="utf-8") as f:
            existing_sections = json.load(f)
            
        # Check if we have substantial content (more than just abstract)
        substantial_sections = [s for s in existing_sections if len(s.get("markdown", "")) > 200]
        if len(substantial_sections) > 2:  # More than just abstract + overview
            logger.info(f"Using existing {len(existing_sections)} sections from cache")
            return existing_sections
    
    # If not enough content, fetch fresh from ar5iv
    logger.info(f"Fetching fresh content for {arxiv_id} from ar5iv")
    try:
        html = _fetch_ar5iv(arxiv_id)
        sections = _extract_sections(html)
        figures = _extract_figures(html)
        
        # Save the fresh content
        with open(sections_file, "w", encoding="utf-8") as f:
            json.dump(sections, f, ensure_ascii=False, indent=2)
            
        figures_file = paper_dir / "figures.json"
        with open(figures_file, "w", encoding="utf-8") as f:
            json.dump(figures, f, ensure_ascii=False, indent=2)
            
        if sections and len(sections) > 2:
            logger.info(f"Successfully fetched {len(sections)} sections")
            return sections
            
    except Exception as e:
        logger.warning(f"Failed to fetch ar5iv content for {arxiv_id}: {e}")
    
    # Final fallback: create enhanced sections from abstract
    index_file = paper_dir / "index.json"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        abstract = metadata.get("abstract", "")
        if abstract:
            logger.info(f"Creating enhanced sections from abstract for {arxiv_id}")
            # Create a minimal BeautifulSoup object for enhanced extraction
            from bs4 import BeautifulSoup
            fake_soup = BeautifulSoup(f"<div><h1 class='title'>Title: {metadata.get('title', '')}</h1><div class='authors'>Authors: {', '.join(metadata.get('authors', []))}</div><blockquote class='abstract'>{abstract}</blockquote></div>", "html.parser")
            enhanced_sections = _enhanced_abstract_extraction(fake_soup, abstract)
            
            # Save enhanced sections
            with open(sections_file, "w", encoding="utf-8") as f:
                json.dump(enhanced_sections, f, ensure_ascii=False, indent=2)
                
            return enhanced_sections
    
    # Absolute fallback
    return [{"title": "Abstract", "html": "", "markdown": "Content could not be retrieved."}]


def _build_deep_dive_prompt(paper_meta: Dict, sections: List[Dict[str, str]], personas: str, duration_minutes: int) -> List[Dict[str, str]]:
    """Build the prompt for deep dive generation."""
    
    # Organize all content
    paper_title = paper_meta.get("title", "Unknown Title")
    authors = ", ".join(paper_meta.get("authors", []))
    abstract = paper_meta.get("abstract", "")
    arxiv_id = paper_meta.get("arxiv_id", "")
    categories = ", ".join(paper_meta.get("categories", []))
    
    # Format sections content
    sections_text = []
    for section in sections:
        title = section.get("title", "")
        content = section.get("markdown", "")
        if content.strip():
            sections_text.append(f"## {title}\n\n{content}")
    
    full_content = "\n\n".join(sections_text)
    
    # Calculate target word count (assuming ~150 words per minute)
    target_words = duration_minutes * 150
    
    messages = [
        {"role": "system", "content": DEEP_DIVE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"HOST PERSONAS: {personas}\n"
            f"PAPER DEEP DIVE ANALYSIS\n\n"
            f"**Title:** {paper_title}\n"
            f"**Authors:** {authors}\n" 
            f"**arXiv ID:** {arxiv_id}\n"
            f"**Categories:** {categories}\n"
            f"**Target Duration:** {duration_minutes} minutes (~{target_words} words)\n\n"
            f"**Abstract:**\n{abstract}\n\n"
            f"**Full Paper Content:**\n{full_content}\n\n"
            "INSTRUCTIONS FOR DEEP DIVE ANALYSIS:\n"
            "- Conduct a comprehensive technical analysis of this single paper\n"
            "- Cover every major aspect: motivation, methodology, experiments, results, analysis\n"
            "- Discuss technical details that expert researchers would find valuable\n"
            "- Analyze experimental design, statistical significance, baseline comparisons\n"
            "- Examine assumptions, limitations, and potential failure modes\n"
            "- Place the work in broader research context and discuss implications\n"
            "- Provide insights on reproducibility, computational requirements, applications\n"
            "- Generate substantial dialogue worthy of a focused deep dive episode\n"
            "- Use the paper's arXiv ID [arXiv:" + arxiv_id + "] when referencing specific claims\n"
            "- Structure the discussion to systematically cover the entire paper\n\n"
            "Format as a natural dialogue. Each speaker turn must start with exactly **Host Name**: (with the bold markdown formatting). This should be expert-level technical discussion."
        )},
    ]
    return messages


@observe(name="generate_deep_dive_script")
def _generate_deep_dive_script(client, paper_meta: Dict, sections: List[Dict[str, str]], personas: str, settings: Settings, duration_minutes: int = 15) -> str:
    """Generate deep dive script for a single paper with Langfuse tracing."""
    messages = _build_deep_dive_prompt(paper_meta, sections, personas, duration_minutes)
    
    # Some models (like gpt-5) don't support temperature parameter
    completion_kwargs = {
        "model": settings.openai_chat_model,
        "messages": messages,
    }
    
    # Only add temperature if not using a model that doesn't support it
    if "gpt-5" not in settings.openai_chat_model.lower():
        completion_kwargs["temperature"] = 0.7
    
    resp = client.chat.completions.create(**completion_kwargs)
    
    return resp.choices[0].message.content


@observe(name="deep_dive_paper")
def deep_dive_paper(settings: Settings, arxiv_id: str, duration_minutes: int = 15, output_run_id: Optional[str] = None) -> str:
    """Generate a deep dive podcast episode for a specific paper."""
    
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not set; deep dive requires OpenAI access.")
    
    # Find the paper across all runs
    paper_info = _find_paper_in_runs(settings, arxiv_id)
    if not paper_info:
        raise ValueError(f"Paper {arxiv_id} not found in any run. Please ingest it first or check the arXiv ID.")
    
    run_id, paper_dir, paper_meta = paper_info
    logger.info(f"Found paper {arxiv_id} in run {run_id}")
    
    # Retrieve the most complete content for the paper
    sections = _retrieve_full_paper_content(arxiv_id, paper_dir, settings)
    logger.info(f"Retrieved {len(sections)} sections for deep dive")
    
    # Set up output directory (either specified run or create new deep-dive run)
    if output_run_id:
        output_run_path = run_dir(settings.data_dir, output_run_id)
    else:
        output_run_id = f"deep-dive-{arxiv_id.replace('/', '_')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        output_run_path = run_dir(settings.data_dir, output_run_id)
    
    scripts_path = scripts_dir(settings.data_dir, output_run_id)
    ensure_dir(scripts_path)
    
    # Generate the deep dive script
    client = get_openai_client(api_key=settings.openai_api_key)
    personas = _format_personas(settings)
    
    logger.info(f"Generating {duration_minutes}-minute deep dive script for {arxiv_id}")
    script_content = _generate_deep_dive_script(client, paper_meta, sections, personas, settings, duration_minutes)
    
    # Save the script
    script_file = scripts_path / f"deep_dive_{arxiv_id.replace('/', '_')}.md"
    script_file.write_text(script_content, encoding="utf-8")
    
    # Create a metadata file for this deep dive
    metadata = {
        "type": "deep_dive",
        "paper": paper_meta,
        "source_run": run_id,
        "generated_at": datetime.now().isoformat(),
        "duration_minutes": duration_minutes,
        "sections_count": len(sections)
    }
    
    metadata_file = output_run_path / "deep_dive_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Deep dive script saved to {script_file}")
    return output_run_id