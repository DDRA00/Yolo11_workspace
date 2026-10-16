from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from scripts import verify_submodule_checks as checks_module
from scripts.verify_submodule_checks import (
    CheckSnapshot,
    SubmoduleCheckError,
    WorkflowSuite,
    _fetch_check_runs,
    _load_policies,
    _successful_evidence,
    verify_submodule_checks,
)


def _write_policy(tmp_path: Path, *, project_path: str = "child") -> Path:
    policy = tmp_path / "checks.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "projects": {
                    project_path: {
                        "repository": "owner/repository",
                        "workflow": ".github/workflows/ci.yml",
                        "required_checks": ["Quality", "Tests"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return policy


def _check(
    name: str,
    conclusion: str | None,
    *,
    app: str = "github-actions",
    run_id: int = 1,
    suite_id: int = 1,
    suite_created_at: str = "2026-08-18T00:00:00Z",
    started_at: str = "2026-08-18T00:00:00Z",
    status: str = "completed",
) -> dict:
    return {
        "id": run_id,
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "details_url": f"https://example.invalid/{name}",
        "app": {"slug": app},
        "check_suite": {"id": suite_id, "created_at": suite_created_at},
        "started_at": started_at,
    }


def _snapshot(
    runs: list[dict],
    *,
    suite_id: int = 1,
    created_at: str = "2026-08-18T00:00:00Z",
    status: str = "completed",
    conclusion: str | None = "success",
) -> CheckSnapshot:
    return CheckSnapshot(
        runs=tuple(runs),
        workflow_suites=(
            WorkflowSuite(
                suite_id=suite_id,
                created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00")),
                status=status,
                conclusion=conclusion,
            ),
        ),
    )


def test_successful_evidence_uses_exact_github_actions_successes() -> None:
    evidence = _successful_evidence(
        [
            _check("Quality", "failure", run_id=1),
            _check("Quality", "success", run_id=2),
            _check("Tests", "success", app="untrusted-app", run_id=3),
        ],
        ("Quality", "Tests"),
    )

    assert evidence == {}


def test_verify_requires_every_named_check_for_the_exact_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _write_policy(tmp_path)
    child = tmp_path / "child"
    child.mkdir()
    revision = "a" * 40
    monkeypatch.setattr(checks_module, "_submodule_head_sha", lambda *_: revision)
    requested: list[tuple[str, str, str]] = []

    def fetch(repository: str, sha: str, workflow: str):
        requested.append((repository, sha, workflow))
        return _snapshot(
            [
                _check("Quality", "success", run_id=1, suite_id=10),
                _check("Tests", "success", run_id=2, suite_id=10),
            ],
            suite_id=10,
        )

    result = verify_submodule_checks(tmp_path, policy, fetch_checks=fetch)

    assert requested == [("owner/repository", revision, ".github/workflows/ci.yml")]
    assert result["projects"][0]["revision"] == revision


def test_verify_rejects_failed_or_missing_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _write_policy(tmp_path)
    (tmp_path / "child").mkdir()
    monkeypatch.setattr(
        checks_module,
        "_submodule_head_sha",
        lambda *_: "b" * 40,
    )

    with pytest.raises(SubmoduleCheckError, match="Tests"):
        verify_submodule_checks(
            tmp_path,
            policy,
            fetch_checks=lambda *_: _snapshot([_check("Quality", "success", run_id=1)]),
        )


@pytest.mark.parametrize(
    ("status", "conclusion"),
    [("completed", "failure"), ("in_progress", None)],
)
def test_newer_failure_or_pending_check_cannot_reuse_stale_success(
    status: str,
    conclusion: str | None,
) -> None:
    evidence = _successful_evidence(
        [
            _check(
                "Quality",
                "success",
                run_id=100,
                suite_id=10,
                started_at="2026-08-18T00:00:00Z",
            ),
            _check(
                "Quality",
                conclusion,
                status=status,
                run_id=1,
                suite_id=10,
                started_at="2026-08-18T00:01:00Z",
            ),
        ],
        ("Quality",),
    )

    assert evidence == {}


def test_required_checks_are_never_combined_across_suites() -> None:
    evidence = _successful_evidence(
        [
            _check("Quality", "success", run_id=10, suite_id=100),
            _check("Tests", "success", run_id=11, suite_id=200),
        ],
        ("Quality", "Tests"),
    )

    assert evidence == {}


def test_parallel_newer_suite_wins_when_older_downstream_check_starts_later() -> None:
    evidence = _successful_evidence(
        [
            _check(
                "Quality",
                "success",
                run_id=1,
                suite_id=10,
                suite_created_at="2026-08-18T00:00:00Z",
                started_at="2026-08-18T00:00:00Z",
            ),
            _check(
                "Tests",
                "failure",
                run_id=4,
                suite_id=10,
                suite_created_at="2026-08-18T00:00:00Z",
                started_at="2026-08-18T00:03:00Z",
            ),
            _check(
                "Quality",
                "success",
                run_id=2,
                suite_id=20,
                suite_created_at="2026-08-18T00:01:00Z",
                started_at="2026-08-18T00:01:00Z",
            ),
            _check(
                "Tests",
                "success",
                run_id=3,
                suite_id=20,
                suite_created_at="2026-08-18T00:01:00Z",
                started_at="2026-08-18T00:02:00Z",
            ),
        ],
        ("Quality", "Tests"),
    )

    assert set(evidence) == {"Quality", "Tests"}


def test_delayed_old_suite_cannot_hide_a_new_pending_suite() -> None:
    evidence = _successful_evidence(
        [
            _check(
                "Quality",
                "success",
                run_id=3,
                suite_id=10,
                suite_created_at="2026-08-18T00:00:00Z",
                started_at="2026-08-18T00:10:00Z",
            ),
            _check(
                "Tests",
                "success",
                run_id=4,
                suite_id=10,
                suite_created_at="2026-08-18T00:00:00Z",
                started_at="2026-08-18T00:10:00Z",
            ),
            _check(
                "Quality",
                None,
                status="queued",
                run_id=2,
                suite_id=20,
                suite_created_at="2026-08-18T00:01:00Z",
                started_at="2026-08-18T00:02:00Z",
            ),
        ],
        ("Quality", "Tests"),
    )

    assert evidence == {}


def test_same_creation_time_requires_every_frontier_suite_to_pass() -> None:
    evidence = _successful_evidence(
        [
            _check("Quality", "success", run_id=1, suite_id=10),
            _check("Tests", "success", run_id=2, suite_id=10),
            _check(
                "Quality",
                None,
                status="in_progress",
                run_id=3,
                suite_id=20,
            ),
        ],
        ("Quality", "Tests"),
    )

    assert evidence == {}


def test_new_workflow_suite_without_required_runs_blocks_stale_success() -> None:
    old_created = datetime.fromisoformat("2026-08-18T00:00:00+00:00")
    new_created = datetime.fromisoformat("2026-08-18T00:01:00+00:00")
    evidence = _successful_evidence(
        [
            _check("Quality", "success", run_id=1, suite_id=10),
            _check("Tests", "success", run_id=2, suite_id=10),
        ],
        ("Quality", "Tests"),
        (
            WorkflowSuite(10, old_created, "completed", "success"),
            WorkflowSuite(20, new_created, "queued", None),
        ),
    )

    assert evidence == {}


def test_new_partial_suite_blocks_an_older_complete_suite() -> None:
    evidence = _successful_evidence(
        [
            _check(
                "Quality",
                "success",
                run_id=1,
                suite_id=10,
                suite_created_at="2026-08-18T00:00:00Z",
                started_at="2026-08-18T00:00:00Z",
            ),
            _check(
                "Tests",
                "success",
                run_id=2,
                suite_id=10,
                suite_created_at="2026-08-18T00:00:00Z",
                started_at="2026-08-18T00:00:00Z",
            ),
            _check(
                "Quality",
                None,
                status="queued",
                run_id=3,
                suite_id=20,
                suite_created_at="2026-08-18T00:01:00Z",
                started_at="2026-08-18T00:01:00Z",
            ),
        ],
        ("Quality", "Tests"),
    )

    assert evidence == {}


def test_one_current_suite_can_replace_older_failed_evidence() -> None:
    evidence = _successful_evidence(
        [
            _check(
                "Quality",
                "failure",
                run_id=1,
                suite_id=10,
                suite_created_at="2026-08-18T00:00:00Z",
                started_at="2026-08-18T00:00:00Z",
            ),
            _check(
                "Tests",
                "failure",
                run_id=2,
                suite_id=10,
                suite_created_at="2026-08-18T00:00:00Z",
                started_at="2026-08-18T00:00:00Z",
            ),
            _check(
                "Quality",
                "success",
                run_id=3,
                suite_id=20,
                suite_created_at="2026-08-18T00:01:00Z",
                started_at="2026-08-18T00:01:00Z",
            ),
            _check(
                "Tests",
                "success",
                run_id=4,
                suite_id=20,
                suite_created_at="2026-08-18T00:01:00Z",
                started_at="2026-08-18T00:01:00Z",
            ),
        ],
        ("Quality", "Tests"),
    )

    assert set(evidence) == {"Quality", "Tests"}


def test_required_check_without_suite_identity_is_rejected() -> None:
    run = _check("Quality", "success")
    del run["check_suite"]

    with pytest.raises(SubmoduleCheckError, match="identity data"):
        _successful_evidence([run], ("Quality",))


def test_policy_rejects_project_path_escape(tmp_path: Path) -> None:
    policy = _write_policy(tmp_path, project_path="../outside")

    with pytest.raises(SubmoduleCheckError, match="escapes"):
        _load_policies(tmp_path.resolve(), policy)


def test_fetch_check_runs_uses_optional_read_only_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, str | None, float]] = []
    response_payload = {
        "total_count": 1,
        "check_runs": [_check("Quality", "success")],
    }
    revision = "c" * 40
    workflow_payload = {
        "total_count": 1,
        "workflow_runs": [
            {
                "id": 10,
                "path": ".github/workflows/ci.yml",
                "check_suite_id": 1,
                "head_sha": revision,
                "created_at": "2026-08-18T00:00:00Z",
                "status": "completed",
                "conclusion": "success",
            }
        ],
    }

    class FakeResponse:
        def __init__(self, payload: dict = response_payload) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, *, timeout: float):
        captured.append(
            (
                request.full_url,
                request.get_header("Authorization"),
                timeout,
            )
        )
        if "/actions/runs?" in request.full_url:
            return FakeResponse(workflow_payload)
        return FakeResponse()

    monkeypatch.setenv("SUBMODULE_CHECKS_TOKEN", "read-only-token")
    monkeypatch.setattr(checks_module.urllib.request, "urlopen", fake_urlopen)

    snapshot = _fetch_check_runs(
        "owner/repository",
        revision,
        ".github/workflows/ci.yml",
    )

    assert list(snapshot.runs) == response_payload["check_runs"]
    assert snapshot.workflow_suites[0].suite_id == 1
    assert captured == [
        (
            "https://api.github.com/repos/owner/repository/commits/"
            f"{revision}/check-runs?filter=all&per_page=100&page=1",
            "Bearer read-only-token",
            20.0,
        ),
        (
            "https://api.github.com/repos/owner/repository/actions/runs?"
            f"head_sha={revision}&per_page=100&page=1",
            "Bearer read-only-token",
            20.0,
        ),
    ]


def test_fetch_check_runs_paginates_until_total_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        {"total_count": 2, "check_runs": [{"name": "Quality"}]},
        {"total_count": 2, "check_runs": [{"name": "Tests"}]},
        {"total_count": 0, "workflow_runs": []},
    ]
    requested_urls: list[str] = []

    class FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, *, timeout: float):
        assert timeout == 20.0
        requested_urls.append(request.full_url)
        return FakeResponse(payloads.pop(0))

    monkeypatch.delenv("SUBMODULE_CHECKS_TOKEN", raising=False)
    monkeypatch.setattr(checks_module.urllib.request, "urlopen", fake_urlopen)

    snapshot = _fetch_check_runs(
        "owner/repository",
        "d" * 40,
        ".github/workflows/ci.yml",
    )

    assert [run["name"] for run in snapshot.runs] == ["Quality", "Tests"]
    assert requested_urls[0].endswith("filter=all&per_page=100&page=1")
    assert requested_urls[1].endswith("filter=all&per_page=100&page=2")
    assert "/actions/runs?" in requested_urls[2]


def test_fetch_check_runs_retries_transient_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return self._payload

    def fake_urlopen(request, *, timeout: float):
        nonlocal attempts
        assert timeout == 20.0
        if "/actions/runs?" in request.full_url:
            return FakeResponse(b'{"total_count": 0, "workflow_runs": []}')
        attempts += 1
        if attempts == 1:
            raise checks_module.urllib.error.URLError("temporary")
        return FakeResponse(b'{"total_count": 0, "check_runs": []}')

    monkeypatch.setattr(checks_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(checks_module.time, "sleep", delays.append)

    snapshot = _fetch_check_runs(
        "owner/repository",
        "e" * 40,
        ".github/workflows/ci.yml",
    )

    assert snapshot == CheckSnapshot(runs=(), workflow_suites=())
    assert attempts == 2
    assert delays == [0.5]


def test_policy_rejects_non_mapping_document(tmp_path: Path) -> None:
    policy = tmp_path / "checks.json"
    policy.write_text("[]", encoding="utf-8")

    with pytest.raises(SubmoduleCheckError, match="schema"):
        _load_policies(tmp_path.resolve(), policy)


def test_policy_rejects_boolean_schema_version(tmp_path: Path) -> None:
    policy = tmp_path / "checks.json"
    policy.write_text(
        json.dumps({"schema_version": True, "projects": {}}),
        encoding="utf-8",
    )

    with pytest.raises(SubmoduleCheckError, match="schema"):
        _load_policies(tmp_path.resolve(), policy)


def test_main_reports_structured_submodule_check_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_verification(*_args, **_kwargs):
        raise SubmoduleCheckError("checks are pending")

    monkeypatch.setattr(checks_module, "verify_submodule_checks", fail_verification)

    exit_code = checks_module.main(["--root", str(tmp_path)])

    payload = json.loads(capsys.readouterr().err)
    assert exit_code == 1
    assert payload == {
        "event": "submodule_exact_sha_checks_failed",
        "error": "checks are pending",
    }
