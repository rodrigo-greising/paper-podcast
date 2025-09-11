from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from ..config import Settings
from ..utils.logging import get_logger
from ..utils.paths import run_dir, audio_dir


logger = get_logger(__name__)


LINE_RE = re.compile(r"^(?P<speaker>[A-Za-z][A-Za-z0-9_ ]+):\s*(?P<text>.+)$")


def _parse_script_lines(text: str) -> List[Tuple[str, str]]:
	lines = []
	for raw in text.splitlines():
		m = LINE_RE.match(raw.strip())
		if not m:
			continue
		lines.append((m.group("speaker"), m.group("text")))
	return lines


def _voice_for_speaker(settings: Settings, speaker: str) -> str:
	s1, s2 = settings.hosts[0], settings.hosts[1]
	if speaker.lower().startswith(s1.name.lower()):
		return s1.voice
	if speaker.lower().startswith(s2.name.lower()):
		return s2.voice
	return s1.voice


def tts_run(settings: Settings, run_id: str) -> None:
	run_path = run_dir(settings.data_dir, run_id)
	scripts_path = run_path / "scripts"
	out_dir = audio_dir(settings.data_dir, run_id)

	if not scripts_path.exists():
		raise FileNotFoundError("Missing scripts; run generate first")

	for script_file in sorted(scripts_path.glob("cluster_*.md")):
		text = script_file.read_text(encoding="utf-8")
		pairs = _parse_script_lines(text)
		seg_dir = out_dir / script_file.stem
		seg_dir.mkdir(parents=True, exist_ok=True)
		for idx, (speaker, line) in enumerate(pairs):
			voice = _voice_for_speaker(settings, speaker)
			wav = seg_dir / f"{idx:04d}_{speaker.replace(' ', '_')}.aiff"
			if settings.use_macos_say:
				# Use macOS say for quick local TTS
				import subprocess
				subprocess.run(["say", "-v", voice, "-o", str(wav), line], check=True)
			logger.info(f"Synthesized {wav}")

	logger.info("TTS synthesis complete")
