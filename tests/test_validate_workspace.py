from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from scripts.validate_workspace import (
    WORKSPACE_ENVIRONMENT_VARIABLE,
    WorkspaceValidationError,
    _configured_import_paths,
    _configured_workspace,
    _project_roots_from_manifest,
    _require_equal_path,
    main,
    validate_workspace,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def test_configured_workspace_restores_absent_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(WORKSPACE_ENVIRONMENT_VARIABLE, raising=False)

    with _configured_workspace(tmp_path):
        assert os.environ[WORKSPACE_ENVIRONMENT_VARIABLE] == str(tmp_path)

    assert WORKSPACE_ENVIRONMENT_VARIABLE not in os.environ


def test_configured_workspace_restores_existing_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(WORKSPACE_ENVIRONMENT_VARIABLE, "existing-workspace")

    with _configured_workspace(tmp_path):
        assert os.environ[WORKSPACE_ENVIRONMENT_VARIABLE] == str(tmp_path)

    assert os.environ[WORKSPACE_ENVIRONMENT_VARIABLE] == "existing-workspace"


def test_configured_import_paths_restores_sys_path(tmp_path: Path) -> None:
    original = list(sys.path)

    with _configured_import_paths(tmp_path):
        assert sys.path[0] == str(tmp_path)

    assert sys.path == original


def test_require_equal_path_rejects_contract_mismatch(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceValidationError, match="training data"):
        _require_equal_path(
            "training data",
            tmp_path / "first",
            tmp_path / "second",
        )


def test_project_roots_are_loaded_from_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "workspace.yaml"
    manifest.write_text(
        "projects:\n  training: train-project\n  inference: infer-project\n",
        encoding="utf-8",
    )

    training_root, inference_root = _project_roots_from_manifest(tmp_path, manifest)

    assert training_root == (tmp_path / "train-project").resolve()
    assert inference_root == (tmp_path / "infer-project").resolve()


def test_project_root_escape_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "workspace.yaml"
    manifest.write_text(
        "projects:\n  training: ../outside\n  inference: inference\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="escapes its root"):
        _project_roots_from_manifest(tmp_path, manifest)


def test_main_reports_missing_manifest_as_structured_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--root", str(tmp_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert payload["event"] == "workspace_contract_invalid"
    assert "manifest is missing" in payload["error"]


def test_checked_out_workspace_contract_and_success_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = validate_workspace(WORKSPACE_ROOT)

    assert result["event"] == "workspace_contract_validated"
    assert Path(result["workspace_root"]) == WORKSPACE_ROOT
    assert Path(result["training_project"]) == WORKSPACE_ROOT / "Yolo11_auto_train"
    assert Path(result["inference_project"]) == WORKSPACE_ROOT / "yolo11_inference"
    assert Path(result["inference_results"]) == WORKSPACE_ROOT / "Result"

    exit_code = main(["--root", str(WORKSPACE_ROOT)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == result
