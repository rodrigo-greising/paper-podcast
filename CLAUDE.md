# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
- **Full pipeline**: `scripts/paper-podcast run --limit 25`
- **Monthly topic-based podcasts**: `scripts/paper-podcast monthly --field cs.AI --year 2024 --month 9`
- **Generate from existing monthly data**: `scripts/paper-podcast monthly-generate --year 2024 --month 9`
- **Step-by-step execution**:
  ```bash
  scripts/paper-podcast ingest --limit 25
  scripts/paper-podcast extract
  scripts/paper-podcast embed
  scripts/paper-podcast cluster
  scripts/paper-podcast generate
  scripts/paper-podcast edit      # NEW: Improve flow and coherence
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
- **Testing**: `pytest` (basic tests in `tests/` directory)
- No formal linting commands configured

## Architecture

### Pipeline Stages
This is an 8-stage podcast generation pipeline with two modes:

**Daily Mode** (original):
1. **Ingest** (`ingest/arxiv_ingest.py`) - Fetches arXiv papers by field (default: cs.AI) with limit
2. **Extract** (`extract/ar5iv_extract.py`) - Processes papers via ar5iv HTML, extracts text and figures
3. **Embed** (`embed/embeddings.py`) - Creates embeddings using OpenAI's text-embedding-3-small
4. **Cluster** (`cluster/topics.py`) - Groups papers by topic using UMAP dimensionality reduction and HDBSCAN clustering (with KMeans fallback)
5. **Generate** (`generate/scripts.py`) - Creates podcast dialogue using OpenAI chat with two personas
6. **Edit** (`generate/edit.py`) - **NEW**: Transforms independent cluster scripts into a cohesive episode with intro, transitions, and thematic connections
7. **TTS** (`tts/say_tts.py`) - Converts text to speech using Kokoro TTS with high-quality neural voices (uses edited script when available)
8. **Assemble** (`assembly/assemble.py`) - Combines audio segments into final MP3

**Monthly Mode** (NEW):
1. **Monthly Ingest** (`ingest/arxiv_ingest.py`) - Fetches ALL papers for a specific month and category (thousands of papers)
2. **Extract** (`extract/ar5iv_extract.py`) - Processes papers via ar5iv HTML, extracts text and figures
3. **Embed** (`embed/embeddings.py`) - Creates embeddings using OpenAI's text-embedding-3-small
4. **Enhanced Cluster** (`cluster/topics.py`) - Groups papers by topic with optimizations for large datasets
5. **Monthly Generate** (`generate/monthly_topics.py`) - Creates separate podcast scripts for each major topic cluster
6. **Monthly TTS** (`generate/monthly_topics.py`) - Generates TTS audio for each topic separately
7. **Monthly Assemble** (`generate/monthly_topics.py`) - Assembles separate MP3 files for each topic

**Monthly Generate Command** (Skip data fetching):
- Use `monthly-generate` when you already have papers, embeddings, and clusters
- Automatically creates subclusters (max 30 papers each) for large topic clusters
- Generates comprehensive technical scripts without content truncation
- Options: `--no-audio` to skip TTS/assembly, `--year`/`--month` to specify date

### Configuration System
- Settings managed via `paper_podcast/config.py` with environment variable overrides
- Required: `OPENAI_API_KEY`
- Key optional settings: `PP_FIELD`, `PP_MAX_PAPERS`, `PP_MIN_PER_SECTION`
- Host personas configurable via `PP_HOST1_VOICE`/`PP_HOST2_VOICE` environment variables (Kokoro TTS voice names)

#### Langfuse Integration (Optional Observability)
- **Purpose**: Automatically traces and monitors all OpenAI API calls for cost tracking, performance analysis, and debugging
- **Setup**: Install `langfuse` package and set environment variables:
  - `LANGFUSE_PUBLIC_KEY`: Your Langfuse project public key
  - `LANGFUSE_SECRET_KEY`: Your Langfuse project secret key
  - `LANGFUSE_HOST`: Langfuse instance URL (optional, defaults to Langfuse Cloud)
- **Self-hosting**: For local Langfuse instances, set `LANGFUSE_HOST=http://localhost:3000`
- **Fallback**: Automatically falls back to vanilla OpenAI client when Langfuse is unavailable
- **Implementation**: See `paper_podcast/observability/langfuse.py` for the client wrapper

### Data Organization
- **Input**: Papers stored in `data/papers/<run_id>/`
- **Output**: Episodes in `data/episodes/<run_id>/` with MP3 and README
- **Assets**: Shared resources in `data/assets/` including Kokoro TTS model files (~350MB)
- Run IDs default to current date (YYYY-MM-DD format)

### CLI Structure
Built with Typer, main commands mirror pipeline stages. Entry point via `scripts/paper-podcast` bash wrapper that activates venv and runs `python -m paper_podcast.cli`.

### Dependencies
Python-only project with scientific computing stack (numpy, pandas, scikit-learn) plus OpenAI API, audio processing (pydub, ffmpeg-python), Kokoro TTS for speech synthesis, web scraping (httpx, beautifulsoup4), advanced clustering (UMAP, HDBSCAN), and optional observability (Langfuse).

### TTS System
- Uses **Kokoro TTS** for high-quality neural speech synthesis
- Model files automatically downloaded to `data/assets/kokoro_models/` on first run
- Supports multiple languages: English (US/UK), French, Italian, Japanese, Chinese
- Default voices: `am_adam` (male) and `af_sarah` (female) for US English
- Cross-platform compatibility (no longer requires macOS)