# Paper Podcast (MVP)

Generate long-form, citation-grounded dialogue podcasts from arXiv papers (cs.AI by default). Ingest → extract (ar5iv) → embed (OpenAI) → cluster → generate (OpenAI) → TTS (macOS say) → assemble (mp3).

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
- TTS uses macOS `say` voices. For higher-quality local TTS, we can add XTTS later.
- Output episode MP3 and simple README are in `data/episodes/<run_id>/`.
