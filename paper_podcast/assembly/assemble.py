from __future__ import annotations

from pathlib import Path

from pydub import AudioSegment

from ..config import Settings
from ..utils.logging import get_logger
from ..utils.paths import run_dir, episode_dir, audio_dir


logger = get_logger(__name__)


def _concat_segments(dir_path: Path) -> AudioSegment:
	segments = []
	for wav in sorted(dir_path.glob("*.aiff")):
		segments.append(AudioSegment.from_file(wav))
	if not segments:
		return AudioSegment.silent(duration=500)
	out = segments[0]
	for s in segments[1:]:
		out += AudioSegment.silent(duration=200)  # short gap
		out += s
	return out


def assemble_run(settings: Settings, run_id: str) -> None:
	run_path = run_dir(settings.data_dir, run_id)
	audio_path = audio_dir(settings.data_dir, run_id)
	final_dir = episode_dir(settings.data_dir, run_id)
	final_dir.mkdir(parents=True, exist_ok=True)

	final = AudioSegment.silent(duration=1000)
	for cluster_dir in sorted(audio_path.glob("cluster_*")):
		seg = _concat_segments(cluster_dir)
		final += seg

	mp3_path = final_dir / "episode.mp3"
	final.export(mp3_path, format="mp3", bitrate="192k")
	logger.info(f"Wrote episode audio to {mp3_path}")

	readme = final_dir / "README.md"
	readme.write_text(
		f"# Episode {run_id}\n\nGenerated with Paper Podcast pipeline.\n\nAudio: {mp3_path.name}\n",
		encoding="utf-8",
	)
	logger.info("Assembly complete")
