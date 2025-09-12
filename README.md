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

## Observability (Langfuse)

- This project can auto-instrument OpenAI calls with Langfuse when available.
- Install `langfuse` in your environment and set:
  - `LANGFUSE_PUBLIC_KEY`
  - `LANGFUSE_SECRET_KEY`
  - `LANGFUSE_HOST` (optional; set to your local instance URL when self-hosting)
- No code changes needed: the generator uses a Langfuse-aware OpenAI client shim in `paper_podcast/observability/langfuse.py`.
- When `langfuse` is not installed or env vars are missing, it falls back to the vanilla OpenAI client.
- TTS uses [Kokoro TTS](https://github.com/nazdridoy/kokoro-tts) with natural-sounding voices. Model files (~350MB) are downloaded to `data/assets/kokoro_models/` on first run.
- Available voices: US English (am_adam, af_sarah, etc.), UK English, French, Italian, Japanese, Chinese.
- Output episode MP3 and simple README are in `data/episodes/<run_id>/`.

## TTS Voice Options
Kokoro TTS supports multiple high-quality voices. Set via environment variables:
- `PP_HOST1_VOICE`: Default `am_adam` (male US English)  
- `PP_HOST2_VOICE`: Default `af_sarah` (female US English)

Popular voice options:
- **US Male**: am_adam, am_echo, am_eric, am_liam, am_michael
- **US Female**: af_sarah, af_bella, af_nova, af_river, af_sky
- **UK**: bf_alice, bf_emma, bm_daniel, bm_george
- **Other languages**: Available in French, Italian, Japanese, Chinese
