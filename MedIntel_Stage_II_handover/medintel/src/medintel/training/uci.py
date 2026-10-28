"""UCI Heart Disease: Janosi, A., Steinbrunn, W., Pfisterer, M. and Detrano, R. (1989). Heart Disease [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X. Licensed CC BY 4.0.

The four "processed" files hold the 14 attributes used in published work. Each row becomes a MedIntel record
with the schema's field names, so training and inference read features through the same code."""

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from medintel.schema import Value

DATASET_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
CITATION = (
    "Janosi, A., Steinbrunn, W., Pfisterer, M. and Detrano, R. (1989). Heart Disease [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X"
)
LICENSE = "CC BY 4.0"
SITES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
EXTRA_FILES = ("heart-disease.names",)
SHA256 = {
    "processed.cleveland.data": "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    "processed.hungarian.data": "d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796",
    "processed.switzerland.data": "834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314",
    "processed.va.data": "e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f",
}
CHEST_PAIN = {1: "typical_angina", 2: "atypical_angina", 3: "non_anginal", 4: "asymptomatic"}
RESTING_ECG = {0: "normal", 1: "st_t_abnormality", 2: "lv_hypertrophy"}
ST_SLOPE = {1: "upsloping", 2: "flat", 3: "downsloping"}
THAL = {3: "normal", 6: "fixed_defect", 7: "reversible_defect"}


@dataclass(frozen=True)
class UCIRecord:
    site: str
    row: int
    x_s: dict[str, Value]
    label: int


def _number(text: str) -> float | None:
    text = text.strip()
    if text in ("", "?"):
        return None
    return float(text)


def _code(text: str, mapping: dict[int, str]) -> str | None:
    value = _number(text)
    return None if value is None else mapping.get(int(value))


def _positive(text: str) -> float | None:
    """Resting blood pressure and cholesterol use 0 for "not measured" in the Swiss and VA files."""
    value = _number(text)
    return None if value is None or value == 0 else value


def _flag(text: str) -> bool | None:
    value = _number(text)
    return None if value is None else value == 1


def record_from_row(cells: list[str], site: str, row: int) -> UCIRecord:
    if len(cells) != 14:
        raise ValueError(f"{site} row {row}: expected 14 values, found {len(cells)}")
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num = cells
    vessels = _number(ca)
    x_s: dict[str, Value] = {
        "age": int(_number(age) or 0),
        "sex": "M" if _number(sex) == 1 else "F",
        "chest_pain_type": _code(cp, CHEST_PAIN),
        "systolic_bp": _positive(trestbps),
        "total_cholesterol": _positive(chol),
        "fasting_glucose_over_120": _flag(fbs),
        "resting_ecg": _code(restecg, RESTING_ECG),
        "max_heart_rate": _number(thalach),
        "exercise_angina": _flag(exang),
        "st_depression": _number(oldpeak),
        "st_slope": _code(slope, ST_SLOPE),
        "major_vessels": None if vessels is None else int(vessels),
        "thal": _code(thal, THAL),
    }
    return UCIRecord(site=site, row=row, x_s=x_s, label=int((_number(num) or 0) > 0))


def parse_site(text: str, site: str) -> list[UCIRecord]:
    lines = [line for line in text.splitlines() if line.strip()]
    return [record_from_row(line.split(","), site, i + 1) for i, line in enumerate(lines)]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(dest: Path, url: str = DATASET_URL) -> dict[str, str]:
    """Download the archive and extract only the known data files. Returns each file's SHA-256."""
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (fixed https URL)
        archive = response.read()
    wanted = {*SITES.values(), *EXTRA_FILES}
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if base not in wanted or base in hashes:
                continue
            data = zf.read(name)
            expected = SHA256.get(base)
            if expected and sha256(data) != expected:
                raise ValueError(f"{base} does not match the recorded SHA-256; the published file has changed")
            (dest / base).write_bytes(data)
            hashes[base] = sha256(data)
    missing = sorted(wanted - set(hashes))
    if missing:
        raise ValueError(f"Archive is missing {', '.join(missing)}")
    return hashes


def load(data_dir: Path) -> dict[str, list[UCIRecord]]:
    out: dict[str, list[UCIRecord]] = {}
    for site, filename in SITES.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run: uv run medintel fetch-data")
        out[site] = parse_site(path.read_text(encoding="utf-8", errors="replace"), site)
    return out


def file_hashes(data_dir: Path) -> dict[str, str]:
    return {filename: sha256((data_dir / filename).read_bytes()) for filename in SITES.values()}
