"""The Python pipeline must reproduce the Stage I concept prototype exactly (tests/golden, regenerated with
tools/export_prototype_golden.js)."""

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import load_sample

from medintel.orchestrator import Orchestrator

GOLDEN = json.loads((Path(__file__).parent / "golden" / "prototype_golden.json").read_text(encoding="utf-8"))
CASES = [(batch["file"], case) for batch in GOLDEN["batches"] for case in batch["cases"]]


def assert_same(actual: Any, expected: Any, path: str = "") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        for key in expected:
            assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_same(a, e, f"{path}[{i}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
    else:
        assert actual == pytest.approx(expected, abs=1e-9), f"{path}: {actual!r} != {expected!r}"


@pytest.mark.parametrize("golden", GOLDEN["batches"], ids=lambda b: b["file"])
def test_validation_matches_prototype(golden: dict[str, Any]) -> None:
    batch = load_sample(golden["file"])
    assert batch.missing_columns == golden["missing_columns"]
    assert len(batch.cases) == golden["valid"]
    assert_same([r.model_dump() for r in batch.rejected], golden["rejected"])


@pytest.mark.parametrize(("file", "golden"), CASES, ids=[case["case"]["case_id"] for _, case in CASES])
def test_case_matches_prototype(file: str, golden: dict[str, Any], prototype_orchestrator: Orchestrator) -> None:
    case = next(c for c in load_sample(file).cases if c.case_id == golden["case"]["case_id"])
    analysis = prototype_orchestrator.run_case(case)

    actual_case = analysis.case.model_dump()
    added = {k: v for k, v in actual_case["x_s"].items() if k not in golden["case"]["x_s"]}
    assert all(v is None for v in added.values()), f"fields the prototype lacks must be empty: {added}"
    actual_case["x_s"] = {k: v for k, v in actual_case["x_s"].items() if k in golden["case"]["x_s"]}
    assert_same(actual_case, golden["case"], "case")
    ml = analysis.ml.model_dump()
    expected_ml = dict(golden["ml"])
    assert [c["key"] for c in ml["importance"]] == expected_ml.pop("importance")
    contributions = [
        {k: c[k] for k in ("key", "label", "unit", "value", "imputed", "contribution")} for c in ml["contributions"]
    ]
    assert_same(contributions, expected_ml.pop("contributions"), "ml.contributions")
    assert_same({k: ml[k] for k in expected_ml}, expected_ml, "ml")
    rules = analysis.rules.model_dump()
    assert_same({k: rules[k] for k in golden["rules"]}, golden["rules"], "rules")
    evidence = [
        {k: e[k] for k in ("source_id", "score", "direction", "matched_on")} for e in analysis.model_dump()["evidence"]
    ]
    assert_same(evidence, golden["evidence"], "evidence")
    context = analysis.context.model_dump()
    assert_same({k: context[k] for k in golden["context"]}, golden["context"], "context")
    assert_same(analysis.fusion.model_dump(), golden["fusion"], "fusion")
    assert analysis.recommendation == golden["recommendation"]
    assert_same(analysis.priority.model_dump(), golden["priority"], "priority")
    assert_same(analysis.report.model_dump(), golden["report"], "report")
