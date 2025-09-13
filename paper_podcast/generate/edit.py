from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

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


EDITING_SYSTEM_PROMPT = """You are an expert podcast editor and producer specializing in technical content. Your task is to take a collection of independent technical discussions and transform them into a cohesive, engaging podcast episode that flows naturally from start to finish.

CRITICAL REQUIREMENTS:
1) CREATE COHESIVE FLOW: Transform disconnected segments into a unified narrative
2) ADD SHOW STRUCTURE: Create engaging intro, smooth transitions, and satisfying conclusion
3) IDENTIFY THEMES: Find overarching patterns and connections between different research topics
4) IMPROVE ENGAGEMENT: Add meta-commentary, forward references, and callbacks to earlier discussions
5) MAINTAIN TECHNICAL DEPTH: Preserve all technical content while improving accessibility
6) CREATE CONTINUITY: Ensure hosts reference previous segments and build on earlier points
7) ADD CONTEXT: Explain how different research areas relate to current trends and each other

OUTPUT FORMAT: Return a complete, edited podcast script with proper host attribution and natural flow."""


def _format_personas(settings: Settings) -> str:
    h1, h2 = settings.hosts[0], settings.hosts[1]
    return (
        f"Host A: {h1.name} — {h1.style}. Host B: {h2.name} — {h2.style}. "
        "Maintain consistent personalities throughout."
    )


def _load_cluster_scripts(scripts_path: Path) -> List[Dict[str, Any]]:
    """Load all cluster scripts and metadata."""
    cluster_data = []

    # Load cluster metadata
    clusters_file = scripts_path.parent / "clusters.json"
    cluster_info = {}
    if clusters_file.exists():
        with open(clusters_file, "r", encoding="utf-8") as f:
            clusters = json.load(f)
            for cluster in clusters:
                cluster_info[cluster["cluster_id"]] = {
                    "label": cluster["label"],
                    "paper_ids": cluster["paper_ids"]
                }

    # Load all cluster scripts
    for script_file in sorted(scripts_path.glob("cluster_*.md")):
        # Extract cluster ID from filename
        cluster_id = script_file.stem.split("_")[1]
        try:
            cluster_id = int(cluster_id)
        except ValueError:
            # Handle cluster_-1.md (noise cluster)
            if cluster_id == "-1":
                cluster_id = -1
            else:
                continue

        content = script_file.read_text(encoding="utf-8")

        cluster_data.append({
            "cluster_id": cluster_id,
            "filename": script_file.name,
            "content": content,
            "label": cluster_info.get(cluster_id, {}).get("label", "Unknown Topic"),
            "paper_ids": cluster_info.get(cluster_id, {}).get("paper_ids", [])
        })

    return cluster_data


def _build_editing_prompt(cluster_data: List[Dict[str, Any]], personas: str, total_duration_minutes: int) -> List[Dict[str, str]]:
    """Build the prompt for editing the complete podcast."""

    # Create overview of all topics
    topic_overview = "\n".join([
        f"- Segment {i+1}: {cluster['label']} ({len(cluster['paper_ids'])} papers)"
        for i, cluster in enumerate(cluster_data)
    ])

    # Combine all script content
    script_segments = []
    for i, cluster in enumerate(cluster_data):
        script_segments.append(f"=== SEGMENT {i+1}: {cluster['label']} ===\n{cluster['content']}")

    combined_scripts = "\n\n" + "="*80 + "\n\n".join(script_segments)

    # Calculate target word count
    target_words = total_duration_minutes * 150  # ~150 words per minute

    messages = [
        {"role": "system", "content": EDITING_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"HOST PERSONAS: {personas}\n"
            f"TARGET EPISODE LENGTH: ~{total_duration_minutes} minutes (~{target_words} words)\n"
            f"NUMBER OF SEGMENTS: {len(cluster_data)}\n\n"
            f"TOPIC OVERVIEW:\n{topic_overview}\n\n"
            "EDITING INSTRUCTIONS:\n"
            "1. Create an engaging 2-3 minute introduction that:\n"
            "   - Welcomes listeners and introduces hosts\n"
            "   - Provides overview of today's research themes\n"
            "   - Sets expectations for the technical depth\n"
            "   - Creates excitement about key findings\n\n"
            "2. Transform the existing segments by:\n"
            "   - Adding smooth transitions between topics\n"
            "   - Creating thematic connections and callbacks\n"
            "   - Having hosts reference earlier discussions\n"
            "   - Building narrative momentum throughout\n\n"
            "3. Add a satisfying 1-2 minute conclusion that:\n"
            "   - Synthesizes key themes from the episode\n"
            "   - Reflects on broader implications\n"
            "   - Thanks listeners and previews future content\n\n"
            "4. Throughout the episode:\n"
            "   - Maintain technical accuracy and depth\n"
            "   - Preserve all citations and paper references\n"
            "   - Keep host personalities consistent\n"
            "   - Add forward references (\"we'll see more of this pattern later\")\n"
            "   - Include meta-commentary on research trends\n\n"
            "EXISTING SCRIPT SEGMENTS TO EDIT:\n"
            f"{combined_scripts}\n\n"
            "Please return the complete edited podcast script with natural flow, engaging structure, and cohesive narrative."
        )},
    ]

    return messages


@observe(name="edit_podcast_script")
def _edit_podcast_script(client, cluster_data: List[Dict[str, Any]], personas: str, settings: Settings) -> str:
    """Edit the complete podcast script with Langfuse tracing."""
    # Calculate total expected duration from all clusters
    total_clusters = len([c for c in cluster_data if c["cluster_id"] != -1])  # Exclude noise cluster
    total_duration = max(20, total_clusters * settings.minutes_per_section + 5)  # Add time for intro/outro

    messages = _build_editing_prompt(cluster_data, personas, total_duration)

    # Check if model supports custom temperature
    chat_params = {
        "model": settings.openai_chat_model,
        "messages": messages,
    }

    # Only add temperature for models that support it
    if not settings.openai_chat_model.startswith("gpt-5"):
        chat_params["temperature"] = 0.6

    resp = client.chat.completions.create(**chat_params)

    return resp.choices[0].message.content


@observe(name="edit_podcast_episode")
def edit_run(settings: Settings, run_id: str) -> None:
    """Edit generated cluster scripts into a cohesive podcast episode."""
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set; skipping editing.")
        return

    run_path = run_dir(settings.data_dir, run_id)
    scripts_path = scripts_dir(settings.data_dir, run_id)

    if not scripts_path.exists():
        logger.warning("No scripts directory found. Run generate first.")
        return

    # Check if we have cluster scripts to edit
    cluster_scripts = list(scripts_path.glob("cluster_*.md"))
    if not cluster_scripts:
        logger.warning("No cluster scripts found to edit.")
        return

    logger.info(f"Found {len(cluster_scripts)} cluster scripts to edit")

    # Load all cluster data
    cluster_data = _load_cluster_scripts(scripts_path)
    if not cluster_data:
        logger.warning("No valid cluster data found.")
        return

    # Filter out noise cluster if it exists
    cluster_data = [c for c in cluster_data if c["cluster_id"] != -1]

    if not cluster_data:
        logger.warning("No non-noise clusters found.")
        return

    # Use Langfuse-instrumented OpenAI client when available
    client = get_openai_client(api_key=settings.openai_api_key or None)
    personas = _format_personas(settings)

    logger.info("Generating edited podcast script...")

    # Generate the edited script
    edited_script = _edit_podcast_script(client, cluster_data, personas, settings)

    # Save the edited script
    edited_script_path = scripts_path / "edited_episode.md"
    edited_script_path.write_text(edited_script, encoding="utf-8")
    logger.info(f"Wrote edited script to {edited_script_path}")

    # Create a metadata file with editing info
    edit_metadata = {
        "original_clusters": len(cluster_data),
        "cluster_topics": [c["label"] for c in cluster_data],
        "total_papers": sum(len(c["paper_ids"]) for c in cluster_data),
        "edited_script_path": str(edited_script_path)
    }

    metadata_path = scripts_path / "edit_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(edit_metadata, f, ensure_ascii=False, indent=2)

    logger.info("Script editing complete")