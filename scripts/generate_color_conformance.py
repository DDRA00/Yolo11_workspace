#!/usr/bin/env python
"""Regenerate the shared color-conformance fixture from both implementations.

The inference runtime (``yolo11_inference/core/stats_color_checker.py``) and
the training-pipeline color gate
(``Yolo11_auto_train/src/picture_tool/color/``) are separate code bases that
judge the same product. A rule that differs between them lets a model clear
the gate and behave differently on the line, and nothing in either repository
can notice on its own.

Rather than couple the two release cadences behind a shared package, they
share a *fixture*: byte-identical copies of
``tests/fixtures/color_conformance.json`` live in both repositories. Each repository's own suite -- which already has
OpenCV -- asserts its implementation matches the cases, and the workspace CI
asserts the two copies are identical. Editing one side's expectations without
the other therefore fails a check instead of drifting quietly.

The contract the fixture encodes:

  * both implementations must agree on whether a region is confidently some
    color at all, and
  * when they are confident, they must name the same color.

Sub-threshold cases stay in with ``confident: false`` rather than being
dropped: "neither is confident" is worth pinning, while which of several
near-zero scores happens to rank highest is tie-breaking noise no contract
should freeze.

Cases where the two legitimately differ are recorded under
``known_divergences`` with the reason, so a difference has to be a deliberate
edit rather than a discovery.

Usage:
    python scripts/generate_color_conformance.py [--check]

``--check`` regenerates in memory and fails if either checked-in copy is out
of date, which is what CI wants.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
INFERENCE = WORKSPACE / "yolo11_inference"
TRAINING = WORKSPACE / "Yolo11_auto_train"

sys.path.insert(0, str(INFERENCE))
sys.path.insert(0, str(TRAINING / "src"))

from core.stats_color_checker import StatsColorChecker  # noqa: E402
from picture_tool.color import color_verifier as trainer  # noqa: E402

COLOR_MODEL = "models/Cable1/A/yolo/color_stats.json"
STATS_PATH = INFERENCE / COLOR_MODEL
DEFAULT_THRESHOLD = 0.35

FIXTURES = (
    INFERENCE / "tests" / "fixtures" / "color_conformance.json",
    TRAINING / "tests" / "fixtures" / "color_conformance.json",
)

#: Each case is a stack of HSV bands, so a case is reproducible from numbers
#: alone -- no image files to keep in sync between two repositories. Bands run
#: across the height by default; ``orientation: vertical`` runs them across the
#: width, and ``height``/``width`` make the patch elongated. Both matter: a
#: per-axis center crop and one derived from ``min(h, w)`` are identical on a
#: square patch, so square-only cases cannot see a sampling-geometry change,
#: and bands that run along the uncropped axis cannot either.
CASES: tuple[tuple[str, dict], ...] = (
    ("pure red", {"bands": [[4, 210, 190]]}),
    ("pure orange", {"bands": [[12, 210, 210]]}),
    ("pure yellow", {"bands": [[28, 190, 220]]}),
    ("pure green", {"bands": [[85, 120, 60]]}),
    ("pure black", {"bands": [[0, 20, 30]]}),
    ("red across the 0/179 seam", {"bands": [[3, 200, 150], [178, 200, 150]]}),
    ("dim red across the seam", {"bands": [[3, 120, 70], [178, 120, 70]]}),
    ("bright red just past the seam", {"bands": [[176, 210, 190]]}),
    ("unsaturated grey carries no evidence", {"bands": [[0, 0, 128]]}),
    ("dark unlit region", {"bands": [[0, 10, 12]]}),
    (
        "red with an orange edge",
        {"bands": [[4, 210, 190], [4, 210, 190], [12, 200, 210]]},
    ),
    (
        "orange with a red edge",
        {"bands": [[12, 210, 210], [12, 210, 210], [4, 200, 190]]},
    ),
    ("yellow leaning orange", {"bands": [[22, 200, 220]]}),
    ("green at the edge of its hue gate", {"bands": [[70, 120, 80]]}),
    ("desaturated red", {"bands": [[4, 60, 190]]}),
    # Elongated, with the bands running along the axis the crop trims most.
    # A wire ROI is this shape, and this is where a per-axis crop and a
    # min(h, w) crop keep different amounts of the end band.
    (
        "wide red strip with an orange end",
        {
            "height": 96,
            "width": 384,
            "orientation": "vertical",
            "bands": [[4, 210, 190], [4, 210, 190], [12, 200, 210]],
        },
    ),
    (
        "tall red strip with an orange end",
        {
            "height": 384,
            "width": 96,
            "bands": [[4, 210, 190], [4, 210, 190], [12, 200, 210]],
        },
    ),
    # A shortcut that returns on the first color it recognizes reports that
    # color even when it is the minority, so a minority band is what makes the
    # shortcut visible at all.
    (
        "yellow minority beside green",
        {
            "bands": [
                [28, 190, 220],
                [28, 190, 220],
                [85, 120, 60],
                [85, 120, 60],
                [85, 120, 60],
            ]
        },
    ),
    (
        "yellow minority beside black",
        {
            "bands": [
                [28, 190, 220],
                [28, 190, 220],
                [0, 20, 30],
                [0, 20, 30],
                [0, 20, 30],
            ]
        },
    ),
    # A dominance override that replaces the winner with a minority color is
    # only observable when that color is in fact the minority.
    (
        "red majority with a green minority",
        {
            "bands": [
                [4, 210, 190],
                [4, 210, 190],
                [4, 210, 190],
                [85, 120, 60],
                [85, 120, 60],
            ]
        },
    ),
)

#: Keyed by case name, because one shared sentence stops being true the moment
#: a second, unrelated difference appears -- and then every divergence carries
#: an explanation that fits none of them.
DIVERGENCE_REASONS: dict[str, str] = {
    "desaturated red": (
        "The training gate takes max(per-color threshold, default) while the "
        "runtime takes the per-color threshold as it stands, so with the "
        "default at 0.35 the gate is stricter than the line for yellow, "
        "orange, red and green. Recorded rather than removed: a gate stricter "
        "than what it gates is the safe direction, but the difference was "
        "undocumented, and pinning it here means either side moving has to be "
        "a deliberate edit."
    ),
}


def render(spec: dict) -> np.ndarray:
    """Build a BGR patch from a band spec.

    ``size`` gives a square patch; ``height``/``width`` give an elongated one,
    and ``orientation`` says which axis the bands run along. Geometry belongs in
    the spec because it is part of the contract: on a square patch a per-axis
    center crop and one derived from ``min(h, w)`` are the same crop, so a
    square-only fixture cannot tell the two apart -- which is how the runtime
    and the gate drifted apart on the elongated ROIs the line measures.
    """
    size = int(spec.get("size", 96))
    height = int(spec.get("height", size))
    width = int(spec.get("width", size))
    bands = spec["bands"]
    vertical = str(spec.get("orientation", "horizontal")) == "vertical"
    hsv = np.zeros((height, width, 3), np.uint8)
    edges = np.linspace(0, width if vertical else height, len(bands) + 1).astype(int)
    for index, band in enumerate(bands):
        low, high = edges[index], edges[index + 1]
        region = hsv[:, low:high] if vertical else hsv[low:high, :]
        region[:, :, 0] = band[0]
        region[:, :, 1] = band[1]
        region[:, :, 2] = band[2]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def inference_verdict(image: np.ndarray) -> tuple[str, bool]:
    result = StatsColorChecker.from_json(STATS_PATH).check(image)
    return result.best_color.casefold(), bool(result.is_ok)


def training_verdict(image: np.ndarray) -> tuple[str, bool]:
    ranges = trainer.load_color_ranges(STATS_PATH)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ratios, _masks, debug = trainer._evaluate_image_improved(image, hsv, lab, ranges)
    color, confidence = trainer._initial_prediction(ratios)
    context = trainer.DecisionContext(
        ratios=ratios, debug_info=debug or {}, hsv_img=hsv, lab_img=lab
    )
    color, confidence = trainer._apply_color_rules(color, confidence, context)
    threshold = trainer._confidence_threshold_for(color, DEFAULT_THRESHOLD)
    return str(color).casefold(), bool(confidence >= threshold)


def build_payload() -> tuple[dict, list[str]]:
    cases: list[dict] = []
    divergences: list[dict] = []
    lines: list[str] = []
    for name, spec in CASES:
        image = render(spec)
        run_color, run_confident = inference_verdict(image)
        gate_color, gate_confident = training_verdict(image)
        agree = run_confident == gate_confident and (
            not run_confident or run_color == gate_color
        )
        lines.append(
            f"{name:<40}"
            f"{run_color + ('/confident' if run_confident else '/unsure'):<20}"
            f"{gate_color + ('/confident' if gate_confident else '/unsure'):<20}"
            f"{'' if agree else '  <- divergence'}"
        )
        if agree:
            entry: dict = {"name": name, "spec": spec, "confident": run_confident}
            if run_confident:
                entry["color"] = run_color
            cases.append(entry)
            continue
        divergences.append(
            {
                "name": name,
                "spec": spec,
                "color": run_color if run_color == gate_color else None,
                "inference_confident": run_confident,
                "training_confident": gate_confident,
                "reason": DIVERGENCE_REASONS.get(name, ""),
            }
        )

    payload = {
        "_comment": (
            "Shared conformance fixture; byte-identical copies live in both "
            "repositories. The color model is embedded rather than referenced "
            "so each repository's suite is self-contained -- a conformance "
            "test that skips when its sibling checkout is absent is half a "
            "guard. Regenerate with scripts/generate_color_conformance.py in "
            "the workspace repository."
        ),
        "cases": cases,
        "color_model": json.loads(STATS_PATH.read_text(encoding="utf-8")),
        "color_model_source": COLOR_MODEL,
        "default_threshold": DEFAULT_THRESHOLD,
        "known_divergences": divergences,
        "schema_version": 2,
    }
    return payload, lines


def serialize(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when a checked-in copy is out of date",
    )
    args = parser.parse_args(argv)

    payload, lines = build_payload()
    text = serialize(payload)
    for line in lines:
        print(line)
    print(
        f"\n{len(payload['cases'])} agreed cases, "
        f"{len(payload['known_divergences'])} recorded divergences"
    )

    # A disagreement nobody has explained is a discovery, not a contract. Refuse
    # to record it in either mode: writing it would turn "the two sides differ
    # here for this reason" into "the two sides differ here", which reads as
    # reviewed and is not.
    unexplained = [
        str(item["name"])
        for item in payload["known_divergences"]
        if not str(item.get("reason", "")).strip()
    ]
    if unexplained:
        print(
            "\nUnexplained divergence: "
            + ", ".join(unexplained)
            + "\nEither fix the side that moved, or add the reason to "
            "DIVERGENCE_REASONS in this script and run it again.",
            file=sys.stderr,
        )
        return 1

    stale = [
        path
        for path in FIXTURES
        if not path.is_file() or path.read_text(encoding="utf-8") != text
    ]
    if args.check:
        if stale:
            print(
                "\nOut of date: "
                + ", ".join(str(path.relative_to(WORKSPACE)) for path in stale)
                + "\nRun: python scripts/generate_color_conformance.py",
                file=sys.stderr,
            )
            return 1
        print("\nBoth fixtures are up to date.")
        return 0

    for path in FIXTURES:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(WORKSPACE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
