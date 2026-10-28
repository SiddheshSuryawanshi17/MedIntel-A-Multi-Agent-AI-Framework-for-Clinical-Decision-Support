import hashlib
import re
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from medintel.agents.context_agent import FixtureWriter
from medintel.agents.fixtures import PROTOTYPE_DRAFTS
from medintel.ingestion import DataIngestionService, IngestionResult, parse_events
from medintel.orchestrator import Orchestrator, create_orchestrator

SAMPLES = Path(str(files("medintel") / "samples"))


def read_sample(name: str) -> str:
    return (SAMPLES / name).read_text(encoding="utf-8")


def load_sample(name: str) -> IngestionResult:
    events = parse_events(read_sample("morning_events.csv")) if name == "morning_batch.csv" else None
    return DataIngestionService().load_text(read_sample(name), name, events)


@pytest.fixture(scope="session")
def morning() -> IngestionResult:
    return load_sample("morning_batch.csv")


@pytest.fixture(scope="session")
def boundary() -> IngestionResult:
    return load_sample("boundary_tests.csv")


@pytest.fixture(scope="session")
def prototype_orchestrator() -> Orchestrator:
    return create_orchestrator(writer=FixtureWriter(PROTOTYPE_DRAFTS))


@pytest.fixture(scope="session")
def orchestrator() -> Orchestrator:
    return create_orchestrator()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class HashEncoder:
    """Bag of words hashed into 256 dimensions, normalized: shared words give a high cosine score."""

    name = "hash-bow-256"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                if len(word) > 3:
                    out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out
