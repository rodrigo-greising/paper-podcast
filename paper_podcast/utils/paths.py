from pathlib import Path
from datetime import date


def ensure_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def run_dir(base: Path, run_id: str | None = None) -> Path:
	if not run_id:
		run_id = date.today().isoformat()
	d = base / "runs" / run_id
	return ensure_dir(d)


def episode_dir(base: Path, run_id: str) -> Path:
	return ensure_dir(base / "episodes" / run_id)


def paper_run_dir(base: Path, run_id: str) -> Path:
	return ensure_dir(run_dir(base, run_id) / "papers")


def scripts_dir(base: Path, run_id: str) -> Path:
	return ensure_dir(run_dir(base, run_id) / "scripts")


def audio_dir(base: Path, run_id: str) -> Path:
	return ensure_dir(run_dir(base, run_id) / "audio")


def vectors_dir(base: Path, run_id: str) -> Path:
	return ensure_dir(run_dir(base, run_id) / "vectors")


def clusters_dir(base: Path, run_id: str) -> Path:
	return ensure_dir(run_dir(base, run_id) / "clusters")
