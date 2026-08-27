"""The two copies of the color-conformance fixture must stay identical.

The inference runtime and the training-pipeline color gate are separate code
bases that judge the same product. A decision rule that moves on one side and
not the other lets a model clear the gate and behave differently on the line,
and neither repository can notice that on its own.

Coupling them behind a shared package would tie two release cadences together;
they share a fixture instead. Each repository's own suite -- which already has
OpenCV -- pins its implementation to the cases. This check is the other half:
it makes editing one side's expectations without the other a failure rather
than a quiet drift. It deliberately needs no numeric dependencies, so the
workspace job stays light.

Regenerate both copies with ``scripts/generate_color_conformance.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parent.parent
COPIES = (
    WORKSPACE / "yolo11_inference" / "tests" / "fixtures" / "color_conformance.json",
    WORKSPACE / "Yolo11_auto_train" / "tests" / "fixtures" / "color_conformance.json",
)

pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in COPIES),
    reason="Both submodules must be checked out for the fixture to be comparable.",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_both_repositories_carry_the_same_conformance_fixture() -> None:
    inference, training = COPIES
    assert _digest(inference) == _digest(training), (
        "The color-conformance fixtures have diverged. One side's expectations "
        "were edited without the other. Regenerate both with "
        "scripts/generate_color_conformance.py, which re-runs the two "
        "implementations and refuses to hide a new disagreement."
    )


def test_fixture_still_covers_something() -> None:
    """A fixture emptied by a bad regeneration would pass every conformance test."""
    payload = json.loads(COPIES[0].read_text(encoding="utf-8"))

    # Schema 2 embeds the color model, so each repository's conformance suite
    # is self-contained instead of skipping when its sibling is absent.
    assert payload["schema_version"] == 2
    assert payload["color_model"]["summary"]
    assert len(payload["cases"]) >= 10
    assert any(case["confident"] for case in payload["cases"])
    assert any(not case["confident"] for case in payload["cases"])


def test_every_recorded_divergence_explains_itself() -> None:
    """A divergence without a reason is indistinguishable from an accident."""
    payload = json.loads(COPIES[0].read_text(encoding="utf-8"))

    for divergence in payload["known_divergences"]:
        assert divergence["reason"].strip(), divergence["name"]
        assert (
            divergence["inference_confident"] != divergence["training_confident"]
            or divergence["color"] is None
        ), f"{divergence['name']} is recorded as a divergence but does not diverge"
