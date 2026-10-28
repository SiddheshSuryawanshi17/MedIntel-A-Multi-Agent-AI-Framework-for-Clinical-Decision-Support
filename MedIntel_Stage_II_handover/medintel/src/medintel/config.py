"""Settings loaded from settings.toml (Section 5.2.1), with environment overrides for paths and roles."""

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from medintel.fusion import FusionSettings

Role = Literal["data_admin", "clinician", "curator", "sys_admin"]
ROLE_LABELS: dict[Role, str] = {
    "data_admin": "Data Administrator",
    "clinician": "Clinician",
    "curator": "Knowledge Curator",
    "sys_admin": "System Administrator",
}


class MLSettings(BaseModel):
    model: str = Field(default="", description="Trained risk model file; empty uses the prototype's demo weights.")


class RetrievalSettings(BaseModel):
    k: int = 5
    min_score: float = 0.55
    backend: Literal["demo", "embedding"] = "demo"
    knowledge_base: str = Field(default="", description="Knowledge base JSON file; empty uses the packaged one.")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index: Literal["numpy", "faiss"] = Field(
        default="numpy", description="Exact inner-product search in numpy, or FAISS IndexFlatIP (same results)."
    )
    strong_score: float = Field(default=0.75, description="Evidence at or above this score counts in conflicts.")


class ContextSettings(BaseModel):
    writer: Literal["template", "claude"] = "template"
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high"] = "low"
    max_tokens: int = 8000
    timeout_s: float = 90.0


class AccessSettings(BaseModel):
    """Roles this server instance may act in. Phase 3 replaces this with authenticated identities."""

    roles: list[Role] = ["data_admin", "clinician", "curator", "sys_admin"]


class Settings(BaseModel):
    fusion: FusionSettings = FusionSettings()
    ml: MLSettings = MLSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    context: ContextSettings = ContextSettings()
    access: AccessSettings = AccessSettings()


def _resolve(path: str, root: Path) -> str:
    return str(root / path) if path and not Path(path).is_absolute() else path


def settings_path(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("MEDINTEL_SETTINGS", "config/settings.toml"))


def load_settings(path: str | Path | None = None) -> Settings:
    candidate = settings_path(path)
    settings = Settings()
    if candidate.is_file():
        with candidate.open("rb") as fh:
            settings = Settings.model_validate(tomllib.load(fh))
        root = candidate.resolve().parent.parent
        settings.ml.model = _resolve(settings.ml.model, root)
        settings.retrieval.knowledge_base = _resolve(settings.retrieval.knowledge_base, root)
    roles = os.environ.get("MEDINTEL_ROLES")
    if roles:
        granted = [r.strip() for r in roles.split(",") if r.strip()]
        settings = settings.model_copy(update={"access": AccessSettings.model_validate({"roles": granted})})
    return settings


def load_env_file(path: str | Path) -> list[str]:
    """Read KEY=VALUE lines (for example ANTHROPIC_API_KEY) without overriding variables already set.
    Returns the names that were loaded, never the values."""
    env = Path(path)
    if not env.is_file():
        return []
    loaded: list[str] = []
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def project_env_file(path: str | Path | None = None) -> Path:
    candidate = settings_path(path)
    return candidate.resolve().parent.parent / ".env" if candidate.is_file() else Path(".env")


def medintel_home() -> Path:
    return Path(os.environ.get("MEDINTEL_HOME", Path.home() / ".medintel")).expanduser()
