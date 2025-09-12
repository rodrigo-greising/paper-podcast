import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv


load_dotenv()


@dataclass
class HostPersona:
	name: str
	voice: str  # Kokoro TTS voice name, e.g., "af_sarah", "am_adam"
	style: str  # short description of persona tone/style


@dataclass
class Settings:
	# Core
	data_dir: Path = Path(os.getenv("PP_DATA_DIR", "/Users/grimoire/paper-podcast/data"))
	assets_dir: Path = Path(os.getenv("PP_ASSETS_DIR", "/Users/grimoire/paper-podcast/data/assets"))
	field_category: str = os.getenv("PP_FIELD", "cs.AI")

	# OpenAI
	openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
	openai_embedding_model: str = os.getenv("PP_EMBED_MODEL", "text-embedding-3-small")
	openai_chat_model: str = os.getenv("PP_CHAT_MODEL", "gpt-4o-mini")

	# Ingest
	max_papers_per_run: int = int(os.getenv("PP_MAX_PAPERS", "100"))

	# Chunking/Embedding
	target_chunk_tokens: int = int(os.getenv("PP_CHUNK_TOKENS", "800"))

	# Generation
	minutes_per_section: int = int(os.getenv("PP_MIN_PER_SECTION", "8"))

	# TTS (legacy setting kept for compatibility)
	use_macos_say: bool = False  # Now using Kokoro TTS instead

	# Personas
	hosts: List[HostPersona] = None

	def __post_init__(self) -> None:
		if self.hosts is None:
			self.hosts = [
				HostPersona(
					name="Alex",
					voice=os.getenv("PP_HOST1_VOICE", "am_adam"),
					style="Curious, precise, likes mechanistic interpretations.",
				),
				HostPersona(
					name="Samantha",
					voice=os.getenv("PP_HOST2_VOICE", "af_sarah"),
					style="Big-picture thinker, connects papers to broader trends.",
				),
			]


def get_settings() -> Settings:
	return Settings()
