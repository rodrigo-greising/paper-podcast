from __future__ import annotations

from pathlib import Path

from pydub import AudioSegment

from ..config import Settings
from ..utils.logging import get_logger
from ..utils.paths import run_dir, episode_dir, audio_dir


logger = get_logger(__name__)


def _concat_segments(dir_path: Path) -> AudioSegment:
	segments = []
	total_input_duration = 0
	for wav in sorted(dir_path.glob("*.wav")):
		segment = AudioSegment.from_file(wav)
		segments.append(segment)
		total_input_duration += len(segment)
	
	if not segments:
		logger.info(f"No audio files found in {dir_path}")
		return AudioSegment.silent(duration=500)
	
	logger.info(f"Found {len(segments)} audio files in {dir_path}, total duration: {total_input_duration/1000:.2f}s")
	
	out = segments[0]
	for s in segments[1:]:
		out += AudioSegment.silent(duration=200)  # short gap
		out += s
	
	logger.info(f"Concatenated {dir_path.name}: {len(out)/1000:.2f}s (including gaps)")
	return out


def assemble_run(settings: Settings, run_id: str) -> None:
	run_path = run_dir(settings.data_dir, run_id)
	audio_path = audio_dir(settings.data_dir, run_id)
	final_dir = episode_dir(settings.data_dir, run_id)
	final_dir.mkdir(parents=True, exist_ok=True)

	final = AudioSegment.silent(duration=1000)
	
	# Handle cluster, deep dive, and edited episode audio directories
	audio_dirs = []
	for pattern in ["cluster_*", "deep_dive_*"]:
		audio_dirs.extend(sorted(audio_path.glob(pattern)))
	# Include single edited episode directory if present
	edited_dir = audio_path / "edited_episode"
	if edited_dir.exists():
		audio_dirs.append(edited_dir)
	logger.info(f"Processing {len(audio_dirs)} audio directories")
	
	for audio_dir_item in audio_dirs:
		seg = _concat_segments(audio_dir_item)
		final += seg

	final_duration_seconds = len(final) / 1000
	logger.info(f"Final episode duration: {final_duration_seconds:.2f}s ({final_duration_seconds/60:.1f} minutes)")

	mp3_path = final_dir / "episode.mp3"
	final.export(mp3_path, format="mp3", bitrate="192k")
	logger.info(f"Wrote episode audio to {mp3_path}")

	readme = final_dir / "README.md"
	readme.write_text(
		f"# Episode {run_id}\n\nGenerated with Paper Podcast pipeline.\n\nAudio: {mp3_path.name}\nDuration: {final_duration_seconds:.2f}s ({final_duration_seconds/60:.1f} minutes)\n",
		encoding="utf-8",
	)
	logger.info("Assembly complete")
