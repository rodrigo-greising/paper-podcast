# Paper Podcast (MVP)

Generate long-form, citation-grounded dialogue podcasts from arXiv papers (cs.AI by default). Ingest → extract (ar5iv) → embed (OpenAI) → cluster → generate (OpenAI) → TTS (Kokoro) → assemble (mp3).

## Quickstart

1. Create and activate venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set environment variables
```bash
export OPENAI_API_KEY=...  # required for embeddings + generation
export PP_FIELD=cs.AI       # optional, default cs.AI
export PP_MAX_PAPERS=50     # optional
export PP_MIN_PER_SECTION=8 # optional, minutes per topic section
export PP_HOST1_VOICE=am_adam   # optional, Kokoro TTS voice for host 1
export PP_HOST2_VOICE=af_sarah  # optional, Kokoro TTS voice for host 2
```

3. Run end-to-end
```bash
scripts/paper-podcast run --limit 25
```
Or stage-by-stage:
```bash
scripts/paper-podcast ingest --limit 25
scripts/paper-podcast extract
scripts/paper-podcast embed
scripts/paper-podcast cluster
scripts/paper-podcast generate
scripts/paper-podcast tts
scripts/paper-podcast assemble
```

## Notes
- Extraction uses ar5iv HTML when available. Figures are kept as HTML in `figures.json` for now.
- Embeddings use `text-embedding-3-small`. Swap in config via env.
- Clustering uses KMeans with a simple k heuristic; can upgrade to HDBSCAN/BERTopic.
- Generation uses OpenAI chat with two personas from `paper_podcast/config.py` (override via env voices).

## Observability (Optional)

### Langfuse Integration
Track and analyze all OpenAI API calls automatically:

1. **Install Langfuse** (optional):
```bash
pip install langfuse
```

2. **Get API keys** from [Langfuse Cloud](https://cloud.langfuse.com) or your self-hosted instance

3. **Set environment variables**:
```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...     # Your project public key
export LANGFUSE_SECRET_KEY=sk-lf-...     # Your project secret key
export LANGFUSE_HOST=https://cloud.langfuse.com  # Optional, defaults to Langfuse Cloud
```

4. **For self-hosted Langfuse**:
```bash
export LANGFUSE_HOST=http://localhost:3000  # Your local instance
```

**Features:**
- Automatic tracing of all OpenAI API calls (embeddings + chat completions)
- Cost tracking and performance metrics
- No code changes required - uses instrumented client wrapper
- Graceful fallback to standard OpenAI client when unavailable

## TTS & Output

- TTS uses [Kokoro TTS](https://github.com/nazdridoy/kokoro-tts) with natural-sounding voices. Model files (~350MB) are downloaded to `data/assets/kokoro_models/` on first run.
- Available voices: US English (am_adam, af_sarah, etc.), UK English, French, Italian, Japanese, Chinese.
- Output episode MP3 and simple README are in `data/episodes/<run_id>/`.

## Security & Configuration

- **API Keys**: Never commit API keys to the repository. Use environment variables only.
- **Data Privacy**: The `data/` directory is excluded from git via `.gitignore`.
- **Environment Files**: `.env` files are also excluded to prevent accidental commits of secrets.
- **Optional Dependencies**: Langfuse is optional and gracefully degrades when unavailable.

## TTS Voice Options
Kokoro TTS supports multiple high-quality voices. Set via environment variables:
- `PP_HOST1_VOICE`: Default `am_adam` (male US English)  
- `PP_HOST2_VOICE`: Default `af_sarah` (female US English)

Popular voice options:
- **US Male**: am_adam, am_echo, am_eric, am_liam, am_michael
- **US Female**: af_sarah, af_bella, af_nova, af_river, af_sky
- **UK**: bf_alice, bf_emma, bm_daniel, bm_george
- **Other languages**: Available in French, Italian, Japanese, Chinese
