from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

from pydub import AudioSegment

from ..config import Settings
from ..utils.logging import get_logger
from ..utils.paths import run_dir, audio_dir


logger = get_logger(__name__)


LINE_RE = re.compile(r"^\*\*(?P<speaker>[A-Za-z][A-Za-z0-9_ ]+):?\*\*:?\s*(?P<text>.+)$")


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


def _synthesize_with_kokoro(text: str, voice: str, output_path: Path, assets_dir: Path) -> None:
	"""Synthesize speech using Kokoro TTS."""
	with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
		tmp_file.write(text)
		tmp_file.flush()
		
		model_path = assets_dir / "kokoro_models" / "kokoro-v1.0.onnx"
		voices_path = assets_dir / "kokoro_models" / "voices-v1.0.bin"
		
		try:
			result = subprocess.run([
				"kokoro-tts", 
				tmp_file.name, 
				str(output_path),
				"--voice", voice,
				"--model", str(model_path),
				"--voices", str(voices_path)
			], check=True, capture_output=True, text=True)
		except subprocess.CalledProcessError as e:
			logger.error(f"Kokoro TTS stdout: {e.stdout}")
			logger.error(f"Kokoro TTS stderr: {e.stderr}")
			raise
		finally:
			Path(tmp_file.name).unlink(missing_ok=True)


def tts_run(settings: Settings, run_id: str) -> None:
	run_path = run_dir(settings.data_dir, run_id)
	scripts_path = run_path / "scripts"
	out_dir = audio_dir(settings.data_dir, run_id)

	if not scripts_path.exists():
		raise FileNotFoundError("Missing scripts; run generate first")

	total_duration_ms = 0
	total_files = 0

	# Handle both cluster scripts and deep dive scripts
	script_patterns = ["cluster_*.md", "deep_dive_*.md"]
	script_files = []
	for pattern in script_patterns:
		script_files.extend(scripts_path.glob(pattern))
	
	for script_file in sorted(script_files):
		text = script_file.read_text(encoding="utf-8")
		pairs = _parse_script_lines(text)
		seg_dir = out_dir / script_file.stem
		seg_dir.mkdir(parents=True, exist_ok=True)
		cluster_duration_ms = 0
		
		for idx, (speaker, line) in enumerate(pairs):
			voice = _voice_for_speaker(settings, speaker)
			wav = seg_dir / f"{idx:04d}_{speaker.replace(' ', '_')}.wav"
			
			try:
				_synthesize_with_kokoro(line, voice, wav, settings.assets_dir)
				
				# Check if the audio file was created and get its duration
				if wav.exists():
					try:
						audio = AudioSegment.from_file(wav)
						duration_ms = len(audio)
						cluster_duration_ms += duration_ms
						total_duration_ms += duration_ms
						total_files += 1
						logger.info(f"Synthesized {wav} ({duration_ms/1000:.2f}s)")
					except Exception as e:
						logger.warning(f"Could not read audio duration for {wav}: {e}")
						logger.info(f"Synthesized {wav}")
				else:
					logger.warning(f"Audio file not created: {wav}")
			except subprocess.CalledProcessError as e:
				logger.error(f"Kokoro TTS failed for {speaker}: {e}")
				continue
		
		logger.info(f"Cluster {script_file.stem}: {cluster_duration_ms/1000:.2f}s total from {len(pairs)} segments")

	logger.info(f"TTS synthesis complete: {total_files} files, {total_duration_ms/1000:.2f}s total duration")
