from datetime import date
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .config import get_settings
from .utils.paths import run_dir

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
	from .tts.say_tts import tts_run
	from .assembly.assemble import assemble_run

	ingest_arxiv(settings, run_id)
	extract_run(settings, run_id)
	embed_run(settings, run_id)
	cluster_run(settings, run_id)
	generate_run(settings, run_id)
	tts_run(settings, run_id)
	assemble_run(settings, run_id)

	out = Path(settings.data_dir) / "episodes" / run_id
	print(f"[bold green]Run completed. Episode in {out}")


if __name__ == "__main__":
	app()
