#!/usr/bin/env python3
"""
Generate TTS audio for existing monthly topic scripts.
This script bypasses the script generation step and only generates audio.
"""

import sys
from pathlib import Path

# Add the paper_podcast module to the path
sys.path.insert(0, str(Path(__file__).parent))

from paper_podcast.config import get_settings
from paper_podcast.generate.monthly_topics import generate_monthly_topic_audio, assemble_monthly_topics
from paper_podcast.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    """Generate audio for all existing monthly topic scripts."""
    settings = get_settings()
    year = 2025
    month = 9
    
    print(f"🎙️  Generating audio for monthly topic scripts ({year}-{month:02d})")
    
    # Check if scripts exist
    scripts_path = Path(settings.data_dir) / "runs" / f"{year}-{month:02d}" / "scripts"
    topic_scripts = list(scripts_path.glob("monthly_topic_*.md"))
    
    if not topic_scripts:
        print(f"❌ No monthly topic scripts found in {scripts_path}")
        return
    
    print(f"📝 Found {len(topic_scripts)} monthly topic scripts")
    for script in topic_scripts:
        print(f"   - {script.name}")
    
    # Generate TTS audio
    print(f"\n🔊 Step 1: Generating TTS audio...")
    generate_monthly_topic_audio(settings, year, month)
    
    # Assemble final podcasts
    print(f"\n🎵 Step 2: Assembling final podcasts...")
    assemble_monthly_topics(settings, year, month)
    
    # Show results
    monthly_topics_dir = Path(settings.data_dir) / "episodes" / "monthly_topics" / f"{year}-{month:02d}"
    if monthly_topics_dir.exists():
        topic_podcasts = list(monthly_topics_dir.glob("*.mp3"))
        print(f"\n✅ Monthly podcasts completed!")
        print(f"📁 Generated {len(topic_podcasts)} topic-based podcasts in {monthly_topics_dir}")
        for podcast in topic_podcasts:
            print(f"   🎧 {podcast.name}")
    else:
        print(f"\n⚠️  No podcasts found in {monthly_topics_dir}")

if __name__ == "__main__":
    main()
