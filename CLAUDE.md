# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
- **Full pipeline**: `scripts/paper-podcast run --limit 25`
- **Step-by-step execution**:
  ```bash
  scripts/paper-podcast ingest --limit 25
  scripts/paper-podcast extract
  scripts/paper-podcast embed
  scripts/paper-podcast cluster
  scripts/paper-podcast generate
  scripts/paper-podcast tts
  scripts/paper-podcast assemble
  ```

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Development
- **Python module execution**: `python -m paper_podcast.cli [command]`
- **Dependencies**: Install via `pip install -r requirements.txt`
- No formal linting/testing commands configured (empty tests/ directory)

## Architecture

### Pipeline Stages
This is a 7-stage podcast generation pipeline:

1. **Ingest** (`ingest/arxiv_ingest.py`) - Fetches arXiv papers by field (default: cs.AI)
2. **Extract** (`extract/ar5iv_extract.py`) - Processes papers via ar5iv HTML, extracts text and figures
3. **Embed** (`embed/embeddings.py`) - Creates embeddings using OpenAI's text-embedding-3-small
4. **Cluster** (`cluster/topics.py`) - Groups papers by topic using KMeans clustering
5. **Generate** (`generate/scripts.py`) - Creates podcast dialogue using OpenAI chat with two personas
6. **TTS** (`tts/say_tts.py`) - Converts text to speech using macOS `say` command
7. **Assemble** (`assembly/assemble.py`) - Combines audio segments into final MP3

### Configuration System
- Settings managed via `paper_podcast/config.py` with environment variable overrides
- Required: `OPENAI_API_KEY`
- Key optional settings: `PP_FIELD`, `PP_MAX_PAPERS`, `PP_MIN_PER_SECTION`
- Host personas configurable via `PP_HOST1_VOICE`/`PP_HOST2_VOICE` environment variables

### Data Organization
- **Input**: Papers stored in `data/papers/<run_id>/`
- **Output**: Episodes in `data/episodes/<run_id>/` with MP3 and README
- **Assets**: Shared resources in `data/assets/`
- Run IDs default to current date (YYYY-MM-DD format)

### CLI Structure
Built with Typer, main commands mirror pipeline stages. Entry point via `scripts/paper-podcast` bash wrapper that activates venv and runs `python -m paper_podcast.cli`.

### Dependencies
Python-only project with scientific computing stack (numpy, pandas, scikit-learn) plus OpenAI API, audio processing (pydub, ffmpeg-python), and web scraping (httpx, beautifulsoup4).