from datetime import date
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .config import get_settings
from .utils.paths import run_dir
from .utils.logging import get_logger

logger = get_logger(__name__)

app = typer.Typer(help="Paper Podcast CLI")


@app.command()
def ingest(
	field: Optional[str] = typer.Option(None, help="arXiv field, e.g., cs.AI"),
	run_id: Optional[str] = typer.Option(None, help="Run identifier (YYYY-MM-DD)"),
	limit: Optional[int] = typer.Option(None, help="Max papers to ingest"),
):
	from .ingest.arxiv_ingest import ingest_arxiv

	settings = get_settings()
	if field:
		settings.field_category = field
	if limit:
		settings.max_papers_per_run = limit

	if not run_id:
		run_id = date.today().isoformat()

	out_dir = run_dir(settings.data_dir, run_id)
	count = ingest_arxiv(settings, run_id)
	print(f"[bold green]Ingested {count} papers into {out_dir}")


@app.command()
def extract(
	run_id: Optional[str] = typer.Option(None, help="Run identifier (YYYY-MM-DD)"),
):
	from .extract.ar5iv_extract import extract_run

	settings = get_settings()
	if not run_id:
		run_id = date.today().isoformat()

	extract_run(settings, run_id)
	print(f"[bold green]Extraction completed for run {run_id}")


@app.command()
def embed(
	run_id: Optional[str] = typer.Option(None, help="Run identifier (YYYY-MM-DD)"),
):
	from .embed.embeddings import embed_run

	settings = get_settings()
	if not run_id:
		run_id = date.today().isoformat()

	embed_run(settings, run_id)
	print(f"[bold green]Embedding completed for run {run_id}")


@app.command()
def cluster(
	run_id: Optional[str] = typer.Option(None, help="Run identifier (YYYY-MM-DD)"),
):
	from .cluster.topics import cluster_run

	settings = get_settings()
	if not run_id:
		run_id = date.today().isoformat()

	cluster_run(settings, run_id)
	print(f"[bold green]Clustering completed for run {run_id}")


@app.command()
def generate(
	run_id: Optional[str] = typer.Option(None, help="Run identifier (YYYY-MM-DD)"),
):
	from .generate.scripts import generate_run

	settings = get_settings()
	if not run_id:
		run_id = date.today().isoformat()

	generate_run(settings, run_id)
	print(f"[bold green]Script generation completed for run {run_id}")


@app.command()
def edit(
	run_id: Optional[str] = typer.Option(None, help="Run identifier (YYYY-MM-DD)"),
):
	from .generate.edit import edit_run

	settings = get_settings()
	if not run_id:
		run_id = date.today().isoformat()

	edit_run(settings, run_id)
	print(f"[bold green]Script editing completed for run {run_id}")


@app.command()
def tts(
	run_id: Optional[str] = typer.Option(None, help="Run identifier (YYYY-MM-DD)"),
):
	from .tts.say_tts import tts_run

	settings = get_settings()
	if not run_id:
		run_id = date.today().isoformat()

	tts_run(settings, run_id)
	print(f"[bold green]TTS completed for run {run_id}")


@app.command()
def assemble(
	run_id: Optional[str] = typer.Option(None, help="Run identifier (YYYY-MM-DD)"),
):
	from .assembly.assemble import assemble_run

	settings = get_settings()
	if not run_id:
		run_id = date.today().isoformat()

	assemble_run(settings, run_id)
	print(f"[bold green]Assembly completed for run {run_id}")


@app.command()
def deep_dive(
	arxiv_id: str = typer.Argument(help="arXiv paper ID (e.g., 2509.09610v1)"),
	duration: Optional[int] = typer.Option(15, help="Target duration in minutes"),
	output_run_id: Optional[str] = typer.Option(None, help="Output run ID (auto-generated if not provided)"),
	generate_audio: bool = typer.Option(False, "--audio", help="Also generate audio (TTS + assembly)"),
):
	"""Generate a deep dive podcast episode for a specific paper."""
	from .generate.deep_dive import deep_dive_paper
	
	settings = get_settings()
	
	try:
		output_run_id = deep_dive_paper(
			settings=settings,
			arxiv_id=arxiv_id,
			duration_minutes=duration,
			output_run_id=output_run_id
		)
		
		print(f"[bold green]Deep dive script generated for {arxiv_id}")
		print(f"[bold blue]Output run ID: {output_run_id}")
		
		if generate_audio:
			from .tts.say_tts import tts_run
			from .assembly.assemble import assemble_run
			
			print("[bold yellow]Generating audio...")
			tts_run(settings, output_run_id)
			assemble_run(settings, output_run_id)
			
			out = Path(settings.data_dir) / "episodes" / output_run_id
			print(f"[bold green]Deep dive episode completed. Audio in {out}")
		else:
			scripts_path = Path(settings.data_dir) / "runs" / output_run_id / "scripts"
			print(f"[bold blue]Script saved in {scripts_path}")
			
	except ValueError as e:
		print(f"[bold red]Error: {e}")
		raise typer.Exit(1)
	except Exception as e:
		logger.error(f"Unexpected error in deep dive: {e}")
		print(f"[bold red]Unexpected error: {e}")
		raise typer.Exit(1)


@app.command()
def run(
	field: Optional[str] = typer.Option(None, help="arXiv field, e.g., cs.AI"),
	run_id: Optional[str] = typer.Option(None, help="Run identifier (YYYY-MM-DD)"),
	limit: Optional[int] = typer.Option(25, help="Max papers to ingest for this run"),
):
	"""End-to-end pipeline for a single run."""
	settings = get_settings()
	if field:
		settings.field_category = field
	if limit:
		settings.max_papers_per_run = limit

	if not run_id:
		run_id = date.today().isoformat()

	from .ingest.arxiv_ingest import ingest_arxiv
	from .extract.ar5iv_extract import extract_run
	from .embed.embeddings import embed_run
	from .cluster.topics import cluster_run
	from .generate.scripts import generate_run
	from .generate.edit import edit_run
	from .tts.say_tts import tts_run
	from .assembly.assemble import assemble_run

	ingest_arxiv(settings, run_id)
	extract_run(settings, run_id)
	embed_run(settings, run_id)
	cluster_run(settings, run_id)
	generate_run(settings, run_id)
	edit_run(settings, run_id)
	tts_run(settings, run_id)
	assemble_run(settings, run_id)

	out = Path(settings.data_dir) / "episodes" / run_id
	print(f"[bold green]Run completed. Episode in {out}")


if __name__ == "__main__":
	app()
