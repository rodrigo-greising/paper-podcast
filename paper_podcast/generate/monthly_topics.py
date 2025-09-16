from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional

from ..observability.langfuse import get_openai_client
from ..config import Settings
from ..utils.logging import get_logger
from ..utils.paths import run_dir, scripts_dir
from .accessibility_levels import get_accessibility_level, get_all_levels

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


MONTHLY_SYSTEM_PROMPT = (
    "You are generating a comprehensive monthly review podcast episode focusing on a specific AI research topic. "
    "This is a deep technical dive for an expert audience covering the month's most significant developments in this topic area. "
    "CRITICAL REQUIREMENTS: "
    "1) MONTHLY CONTEXT: Frame this as 'This Month in [Topic]' - discuss trends, breakthroughs, and developments from the entire month. "
    "2) TECHNICAL DEPTH: Cover methodologies, architectures, experimental results, and algorithmic innovations in detail. "
    "3) COMPREHENSIVE COVERAGE: Discuss multiple papers that represent different approaches or sub-areas within the topic. "
    "4) TREND ANALYSIS: Identify patterns, emerging directions, and how this month's work builds on or diverges from previous research. "
    "5) CRITICAL EVALUATION: Analyze strengths, limitations, and potential impact of the research. "
    "6) CITATIONS: Include precise inline citations [arXiv:ID] after each technical claim. "
    "7) STRUCTURE: Organize by themes or approaches, not just individual papers. Show how papers relate to each other. "
    "8) LENGTH: Generate substantial content - aim for 15-25 minutes of detailed technical discussion. "
    "9) EXPERT PERSPECTIVE: Assume listeners are domain experts who want nuanced technical analysis. "
    "The hosts should demonstrate deep expertise and provide insights that only experienced researchers would appreciate."
)


def _format_personas(settings: Settings) -> str:
    h1, h2 = settings.hosts[0], settings.hosts[1]
    return (
        f"Host A: {h1.name} — {h1.style}. Host B: {h2.name} — {h2.style}. "
        "Alternate turns. Avoid filler."
    )


def _create_subclusters(cluster: dict, chunk_rows: List[dict], max_papers_per_subcluster: int = 30) -> List[dict]:
    """Create subclusters for large clusters to enable multiple episodes."""
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    
    # Group content by paper
    papers_content = {}
    for r in chunk_rows:
        paper_id = r['paper_id']
        if paper_id not in papers_content:
            papers_content[paper_id] = []
        papers_content[paper_id].append(f"{r['section_title']}: {r['text']}")
    
    paper_ids = list(papers_content.keys())
    
    if len(paper_ids) <= max_papers_per_subcluster:
        # Small enough for single episode
        return [{
            'subcluster_id': 0,
            'label': cluster['label'],
            'paper_ids': paper_ids,
            'papers_content': papers_content
        }]
    
    # Need to create subclusters
    logger.info(f"Cluster {cluster['cluster_id']} has {len(paper_ids)} papers, creating subclusters")
    
    # Extract embeddings for subclustering
    df = pd.DataFrame(chunk_rows)
    embeddings = np.vstack(df['embedding'].tolist())
    
    # Determine number of subclusters
    n_subclusters = max(2, (len(paper_ids) + max_papers_per_subcluster - 1) // max_papers_per_subcluster)
    
    # Create subclusters using KMeans
    kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
    subcluster_labels = kmeans.fit_predict(embeddings)
    
    # Group papers by subcluster
    subclusters = []
    for subcluster_id in range(n_subclusters):
        subcluster_mask = subcluster_labels == subcluster_id
        subcluster_paper_ids = df[subcluster_mask]['paper_id'].unique().tolist()
        
        if len(subcluster_paper_ids) == 0:
            continue
            
        # Create subcluster label based on top terms
        from sklearn.feature_extraction.text import TfidfVectorizer
        subcluster_texts = []
        for paper_id in subcluster_paper_ids:
            subcluster_texts.extend(papers_content[paper_id])
        
        if subcluster_texts:
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")
            X = vectorizer.fit_transform(subcluster_texts)
            terms = np.array(vectorizer.get_feature_names_out())
            centroid = np.asarray(X.mean(axis=0)).ravel()
            top_idx = centroid.argsort()[::-1][:3]
            subcluster_label = ", ".join(terms[top_idx])
        else:
            subcluster_label = f"{cluster['label']} - Part {subcluster_id + 1}"
        
        subclusters.append({
            'subcluster_id': subcluster_id,
            'label': subcluster_label,
            'paper_ids': subcluster_paper_ids,
            'papers_content': {pid: papers_content[pid] for pid in subcluster_paper_ids}
        })
    
    logger.info(f"Created {len(subclusters)} subclusters for cluster {cluster['cluster_id']}")
    return subclusters


def _build_monthly_prompt_for_cluster(cluster: dict, chunk_rows: List[dict], personas: str,
                                    year: int, month: int, topic_name: str,
                                    accessibility_level: int = 1) -> List[dict]:
    """Build prompt for monthly topic-based podcast generation with accessibility level support."""

    # Get accessibility level configuration
    level_config = get_accessibility_level(accessibility_level)

    # Group content by paper for better organization
    papers_content = {}
    for r in chunk_rows:
        paper_id = r['paper_id']
        if paper_id not in papers_content:
            papers_content[paper_id] = []
        papers_content[paper_id].append(f"{r['section_title']}: {r['text']}")

    # Build structured context with paper content
    context_parts = []
    for paper_id, sections in papers_content.items():
        # Use all sections without truncation
        paper_context = f"[arXiv:{paper_id}]\n" + "\n\n".join(sections)
        context_parts.append(paper_context)

    context_text = "\n\n---\n\n".join(context_parts)

    # Calculate expected word count based on accessibility level
    word_targets = {1: 3000, 2: 2500, 3: 2000, 4: 1500, 5: 1200}
    target_words = word_targets[accessibility_level]

    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    month_name = month_names[month - 1]

    messages = [
        {"role": "system", "content": level_config.system_prompt},
        {"role": "user", "content": (
            f"HOST PERSONAS: {personas}\n"
            f"ACCESSIBILITY LEVEL: {level_config.name} (Level {accessibility_level})\n"
            f"TARGET AUDIENCE: {level_config.target_audience}\n"
            f"MONTHLY TOPIC: {topic_name}\n"
            f"MONTH: {month_name} {year}\n"
            f"TARGET LENGTH: ~{target_words} words\n"
            f"NUMBER OF PAPERS ANALYZED: {len(papers_content)}\n"
            "TECHNICAL CONTENT FOR MONTHLY ANALYSIS:\n"
            f"{context_text}\n\n"
            "INSTRUCTIONS:\n"
            f"- Tailor content for {level_config.target_audience}\n"
            "- Generate a comprehensive monthly review, not just individual paper summaries\n"
            "- Identify and discuss major themes, trends, and breakthroughs from this month\n"
            "- Show how different papers relate to each other and build on common themes\n"
            "- Frame discussions in the context of 'This Month in [Topic]' - what happened, why it matters, where it's heading\n"
            f"- Adjust technical depth and vocabulary appropriate for {level_config.name} level\n"
            "- Include citations and explanations as specified in the system prompt\n\n"
            "Format as a structured dialogue with **Host Name**: before each statement."
        )},
    ]
    return messages


@observe(name="generate_monthly_topic_script")
def _generate_monthly_topic_script(client, cluster: dict, rows: List[dict], personas: str,
                                  settings: Settings, year: int, month: int,
                                  accessibility_level: int = 1) -> str:
    """Generate script for a monthly topic cluster with Langfuse tracing."""
    topic_name = cluster['label']
    messages = _build_monthly_prompt_for_cluster(cluster, rows, personas, year, month, topic_name, accessibility_level)
    
    # Check if model supports custom temperature
    chat_params = {
        "model": settings.openai_chat_model,
        "messages": messages,
    }

    # Only add temperature for models that support it
    if not settings.openai_chat_model.startswith("gpt-5"):
        chat_params["temperature"] = 0.7

    resp = client.chat.completions.create(**chat_params)
    
    return resp.choices[0].message.content


@observe(name="generate_monthly_topic_podcasts")
def generate_monthly_topics(settings: Settings, year: int, month: int,
                           accessibility_levels: Optional[List[int]] = None) -> None:
    """Generate separate podcast episodes for each major topic cluster from a monthly dataset."""
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set; skipping generation.")
        return

    # Default to all accessibility levels if none specified
    if accessibility_levels is None:
        accessibility_levels = [1, 2, 3, 4, 5]

    month_id = f"{year}-{month:02d}"
    run_path = run_dir(settings.data_dir, month_id)
    vectors_path = run_path / "vectors.parquet"
    clusters_path = run_path / "clusters.json"
    scripts_path = scripts_dir(settings.data_dir, month_id)

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

    # Filter clusters to only include substantial ones (more than 5 papers)
    substantial_clusters = [c for c in clusters if len(c["paper_ids"]) >= 5]

    if not substantial_clusters:
        logger.warning("No substantial clusters found (minimum 5 papers per cluster).")
        return

    logger.info(f"Generating monthly topic podcasts for {len(substantial_clusters)} substantial clusters")
    logger.info(f"Creating {len(accessibility_levels)} accessibility levels: {accessibility_levels}")

    episode_count = 0
    for cluster in substantial_clusters:
        paper_ids = set(cluster["paper_ids"])
        rows = df[df["paper_id"].isin(paper_ids)].sort_values(["paper_id", "chunk_index"]).to_dict("records")
        if not rows:
            continue

        # Create subclusters for large clusters
        subclusters = _create_subclusters(cluster, rows)

        for subcluster in subclusters:
            # Get rows for this subcluster
            subcluster_paper_ids = set(subcluster["paper_ids"])
            subcluster_rows = [r for r in rows if r["paper_id"] in subcluster_paper_ids]

            if not subcluster_rows:
                continue

            # Generate scripts for each accessibility level
            for accessibility_level in accessibility_levels:
                level_config = get_accessibility_level(accessibility_level)

                # Generate script for this subcluster and accessibility level
                text = _generate_monthly_topic_script(client, subcluster, subcluster_rows, personas,
                                                    settings, year, month, accessibility_level)

                # Create topic-specific filename with accessibility level
                topic_slug = subcluster['label'].lower().replace(' ', '_').replace(',', '').replace(':', '')[:50]
                if len(subclusters) > 1:
                    # Add part number for subclusters
                    out = scripts_path / f"monthly_topic_{cluster['cluster_id']:02d}_{subcluster['subcluster_id']:02d}_{topic_slug}_{level_config.file_suffix}.md"
                else:
                    out = scripts_path / f"monthly_topic_{cluster['cluster_id']:02d}_{topic_slug}_{level_config.file_suffix}.md"

                out.write_text(text, encoding="utf-8")
                logger.info(f"Wrote {level_config.name} level script: {out}")
                episode_count += 1

    logger.info(f"Monthly topic script generation complete for {month_id} - Generated {episode_count} total scripts across {len(accessibility_levels)} accessibility levels")


def generate_monthly_topic_audio(settings: Settings, year: int, month: int,
                                accessibility_levels: Optional[List[int]] = None) -> None:
    """Generate TTS audio for monthly topic scripts."""
    from ..tts.say_tts import tts_run

    # Default to all accessibility levels if none specified
    if accessibility_levels is None:
        accessibility_levels = [1, 2, 3, 4, 5]

    month_id = f"{year}-{month:02d}"
    scripts_path = scripts_dir(settings.data_dir, month_id)

    # Find all monthly topic scripts
    topic_scripts = list(scripts_path.glob("monthly_topic_*.md"))

    if not topic_scripts:
        logger.warning(f"No monthly topic scripts found in {scripts_path}")
        return

    logger.info(f"Generating TTS audio for {len(topic_scripts)} monthly topic scripts")

    for script_path in topic_scripts:
        # Create a temporary run_id for this topic
        topic_name = script_path.stem
        topic_run_id = f"{month_id}_{topic_name}"

        # Copy the script to the topic-specific run directory
        topic_scripts_dir = scripts_dir(settings.data_dir, topic_run_id)
        topic_scripts_dir.mkdir(parents=True, exist_ok=True)

        # Copy the script
        topic_script_path = topic_scripts_dir / "edited_episode.md"
        topic_script_path.write_text(script_path.read_text(encoding="utf-8"), encoding="utf-8")

        # Generate TTS for this topic
        logger.info(f"Generating TTS for topic: {topic_name}")
        tts_run(settings, topic_run_id)


def assemble_monthly_topics(settings: Settings, year: int, month: int,
                          accessibility_levels: Optional[List[int]] = None) -> None:
    """Assemble individual topic podcasts into separate MP3 files."""
    from ..assembly.assemble import assemble_run

    # Default to all accessibility levels if none specified
    if accessibility_levels is None:
        accessibility_levels = [1, 2, 3, 4, 5]

    month_id = f"{year}-{month:02d}"
    scripts_path = scripts_dir(settings.data_dir, month_id)

    # Find all monthly topic scripts
    topic_scripts = list(scripts_path.glob("monthly_topic_*.md"))

    if not topic_scripts:
        logger.warning(f"No monthly topic scripts found in {scripts_path}")
        return

    logger.info(f"Assembling audio for {len(topic_scripts)} monthly topic podcasts")

    for script_path in topic_scripts:
        topic_name = script_path.stem
        topic_run_id = f"{month_id}_{topic_name}"

        # Assemble audio for this topic
        logger.info(f"Assembling audio for topic: {topic_name}")
        assemble_run(settings, topic_run_id)

        # Move the final episode to a monthly topics directory with organized structure
        from ..utils.paths import episode_dir
        topic_episode_dir = episode_dir(settings.data_dir, topic_run_id)

        # Organize by accessibility level
        level_suffix = None
        for level in accessibility_levels:
            level_config = get_accessibility_level(level)
            if topic_name.endswith(f"_{level_config.file_suffix}"):
                level_suffix = level_config.file_suffix
                level_name = level_config.name.lower()
                break

        if level_suffix:
            monthly_topics_dir = Path(settings.data_dir) / "episodes" / "monthly_topics" / month_id / level_name
        else:
            monthly_topics_dir = Path(settings.data_dir) / "episodes" / "monthly_topics" / month_id / "unknown"

        monthly_topics_dir.mkdir(parents=True, exist_ok=True)

        if topic_episode_dir.exists():
            # Move episode files to monthly topics directory
            for file_path in topic_episode_dir.iterdir():
                new_path = monthly_topics_dir / f"{topic_name}{file_path.suffix}"
                file_path.rename(new_path)
                logger.info(f"Moved {file_path.name} to {new_path}")

            # Remove empty topic episode directory
            topic_episode_dir.rmdir()

    logger.info(f"Monthly topic assembly complete for {month_id}")
