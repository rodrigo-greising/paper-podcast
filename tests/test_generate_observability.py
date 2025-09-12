import json
import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure repo root is on sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from paper_podcast.config import Settings, HostPersona
from paper_podcast.generate.scripts import _format_personas, generate_run
from paper_podcast.utils.paths import run_dir


def test_format_personas_default_names():
    s = Settings()
    personas = _format_personas(s)
    # Defaults are Alex and Samantha per config.py
    assert "Alex" in personas
    assert "Samantha" in personas


def test_generate_run_writes_scripts_and_uses_prompt(tmp_path, monkeypatch):
    # Prepare minimal data for a run
    run_id = "testrun"
    settings = Settings(
        data_dir=tmp_path,
        openai_api_key="test-key",
        minutes_per_section=2,  # small to simplify assertions
        hosts=[
            HostPersona(name="A", voice="am_adam", style="Analytical"),
            HostPersona(name="B", voice="af_sarah", style="Big picture"),
        ],
    )

    run_path = run_dir(settings.data_dir, run_id)
    # Create vectors.parquet
    df = pd.DataFrame(
        [
            {
                "paper_id": "1234.5678",
                "chunk_index": 0,
                "section_title": "Introduction",
                "text": "We introduce a new model.",
            },
            {
                "paper_id": "1234.5678",
                "chunk_index": 1,
                "section_title": "Method",
                "text": "The method uses attention.",
            },
        ]
    )
    vectors_path = run_path / "vectors.parquet"
    df.to_parquet(vectors_path)

    # Create clusters.json
    clusters = [
        {
            "cluster_id": 1,
            "label": "Transformer Advances",
            "paper_ids": ["1234.5678"],
        }
    ]
    clusters_path = run_path / "clusters.json"
    clusters_path.write_text(json.dumps(clusters), encoding="utf-8")

    # Stub OpenAI client to capture the prompt and return a fixed response
    captured = {}

    class DummyCompletions:
        def create(self, model, messages, temperature):
            captured["model"] = model
            captured["messages"] = messages
            captured["temperature"] = temperature

            class _Msg:
                def __init__(self, content):
                    self.content = content

            class _Choice:
                def __init__(self, content):
                    self.message = _Msg(content)

            class _Resp:
                def __init__(self, content):
                    self.choices = [_Choice(content)]

            return _Resp("SCRIPT CONTENT")

    class DummyClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=DummyCompletions())

    def fake_get_openai_client(api_key=None):
        return DummyClient()

    monkeypatch.setattr(
        "paper_podcast.generate.scripts.get_openai_client", fake_get_openai_client
    )

    # Run generation
    generate_run(settings, run_id)

    # Assert a script file is produced
    out_file = tmp_path / "runs" / run_id / "scripts" / "cluster_01.md"
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == "SCRIPT CONTENT"

    # Inspect captured prompt to ensure key context exists
    msgs = captured.get("messages")
    assert msgs and msgs[0]["role"] == "system"
    assert "Transformer Advances" in msgs[1]["content"]  # cluster label
    assert "NUMBER OF PAPERS: 1" in msgs[1]["content"]
    assert "[arXiv:1234.5678]" in msgs[1]["content"]  # per-paper context header
    assert "TARGET LENGTH: ~2 minutes" in msgs[1]["content"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
