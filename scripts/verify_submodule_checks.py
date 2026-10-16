"""Require successful GitHub checks for each checked-out submodule commit."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


_GITHUB_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_WORKFLOW_PATH = re.compile(r"^\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml$")
_SUCCESS = "success"


class SubmoduleCheckError(RuntimeError):
    """Raised when exact-revision CI evidence is absent or unsuccessful."""


@dataclass(frozen=True)
class SubmoduleCheckPolicy:
    path: Path
    repository: str
    workflow: str
    required_checks: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowSuite:
    suite_id: int
    created_at: datetime
    status: str
    conclusion: str | None


@dataclass(frozen=True)
class CheckSnapshot:
    runs: tuple[Mapping[str, Any], ...]
    workflow_suites: tuple[WorkflowSuite, ...]


CheckFetcher = Callable[[str, str, str], CheckSnapshot]


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(".github/submodule-required-checks.json"),
    )
    return parser.parse_args(arguments)


def _load_policies(root: Path, policy_path: Path) -> tuple[SubmoduleCheckPolicy, ...]:
    resolved_policy = (
        policy_path if policy_path.is_absolute() else root / policy_path
    ).resolve()
    try:
        payload = json.loads(resolved_policy.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SubmoduleCheckError(
            f"Submodule check policy is unreadable: {error}"
        ) from error
    schema_version = (
        payload.get("schema_version") if isinstance(payload, Mapping) else None
    )
    projects = payload.get("projects") if isinstance(payload, Mapping) else None
    if (
        type(schema_version) is not int
        or schema_version != 2
        or not isinstance(projects, Mapping)
    ):
        raise SubmoduleCheckError("Submodule check policy schema is invalid.")

    policies: list[SubmoduleCheckPolicy] = []
    for raw_path, raw_policy in projects.items():
        if not isinstance(raw_path, str) or not isinstance(raw_policy, Mapping):
            raise SubmoduleCheckError("Submodule check policy entry is invalid.")
        project_path = Path(raw_path)
        resolved_project = (root / project_path).resolve()
        if (
            project_path.is_absolute()
            or resolved_project == root
            or not resolved_project.is_relative_to(root)
        ):
            raise SubmoduleCheckError(
                f"Submodule check path escapes the workspace: {raw_path}"
            )
        repository = raw_policy.get("repository")
        workflow = raw_policy.get("workflow")
        required_checks = raw_policy.get("required_checks")
        if not isinstance(repository, str) or not _GITHUB_REPOSITORY.fullmatch(
            repository
        ):
            raise SubmoduleCheckError(
                f"Invalid GitHub repository for submodule {raw_path}."
            )
        if not isinstance(workflow, str) or not _WORKFLOW_PATH.fullmatch(workflow):
            raise SubmoduleCheckError(
                f"Invalid GitHub Actions workflow for submodule {raw_path}."
            )
        if (
            not isinstance(required_checks, list)
            or not required_checks
            or any(
                not isinstance(check, str) or not check.strip()
                for check in required_checks
            )
            or len(set(required_checks)) != len(required_checks)
        ):
            raise SubmoduleCheckError(
                f"Required checks are invalid for submodule {raw_path}."
            )
        policies.append(
            SubmoduleCheckPolicy(
                path=project_path,
                repository=repository,
                workflow=workflow,
                required_checks=tuple(required_checks),
            )
        )
    if not policies:
        raise SubmoduleCheckError("Submodule check policy has no projects.")
    return tuple(policies)


def _submodule_head_sha(root: Path, relative_path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root / relative_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SubmoduleCheckError(
            f"Unable to read submodule revision for {relative_path}: {error}"
        ) from error
    revision = completed.stdout.strip().lower()
    if not _GIT_SHA.fullmatch(revision):
        raise SubmoduleCheckError(
            f"Submodule revision is invalid for {relative_path}: {revision!r}"
        )
    return revision


def _fetch_check_runs(
    repository: str,
    revision: str,
    workflow: str,
) -> CheckSnapshot:
    owner, name = repository.split("/", maxsplit=1)
    encoded_repository = "/".join(
        urllib.parse.quote(segment, safe="") for segment in (owner, name)
    )
    runs: list[Mapping[str, Any]] = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{encoded_repository}/commits/"
            f"{revision}/check-runs?filter=all&per_page=100&page={page}"
        )
        request = urllib.request.Request(url, headers=_github_headers())
        payload = _read_github_json(request, repository, revision)
        page_runs = payload.get("check_runs") if isinstance(payload, Mapping) else None
        total_count = (
            payload.get("total_count") if isinstance(payload, Mapping) else None
        )
        if (
            not isinstance(page_runs, list)
            or type(total_count) is not int
            or total_count < 0
        ):
            raise SubmoduleCheckError(
                f"GitHub returned invalid check data for {repository}@{revision}."
            )
        if any(not isinstance(run, Mapping) for run in page_runs):
            raise SubmoduleCheckError(
                f"GitHub returned an invalid check run for {repository}@{revision}."
            )
        runs.extend(page_runs)
        if len(runs) >= total_count:
            break
        if not page_runs:
            raise SubmoduleCheckError(
                f"GitHub check pagination stopped early for {repository}@{revision}."
            )
        page += 1
    return CheckSnapshot(
        runs=tuple(runs),
        workflow_suites=_fetch_workflow_suites(
            repository,
            revision,
            workflow,
            encoded_repository,
        ),
    )


def _fetch_workflow_suites(
    repository: str,
    revision: str,
    workflow: str,
    encoded_repository: str,
) -> tuple[WorkflowSuite, ...]:
    suites: list[WorkflowSuite] = []
    seen_suite_ids: set[int] = set()
    fetched_count = 0
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{encoded_repository}/actions/runs?"
            f"head_sha={revision}&per_page=100&page={page}"
        )
        request = urllib.request.Request(url, headers=_github_headers())
        payload = _read_github_json(request, repository, revision)
        page_runs = (
            payload.get("workflow_runs") if isinstance(payload, Mapping) else None
        )
        total_count = (
            payload.get("total_count") if isinstance(payload, Mapping) else None
        )
        if (
            not isinstance(page_runs, list)
            or type(total_count) is not int
            or total_count < 0
            or any(not isinstance(run, Mapping) for run in page_runs)
        ):
            raise SubmoduleCheckError(
                f"GitHub returned invalid workflow data for {repository}@{revision}."
            )
        for run in page_runs:
            if run.get("path") != workflow:
                continue
            run_id = run.get("id")
            suite_id = run.get("check_suite_id")
            head_sha = run.get("head_sha")
            created_at = run.get("created_at")
            status = run.get("status")
            conclusion = run.get("conclusion")
            if (
                type(run_id) is not int
                or run_id <= 0
                or type(suite_id) is not int
                or suite_id <= 0
                or not isinstance(head_sha, str)
                or head_sha.lower() != revision
                or not isinstance(created_at, str)
                or not isinstance(status, str)
                or not status
                or (conclusion is not None and not isinstance(conclusion, str))
                or suite_id in seen_suite_ids
            ):
                raise SubmoduleCheckError(
                    f"GitHub returned invalid CI workflow identity for "
                    f"{repository}@{revision}."
                )
            seen_suite_ids.add(suite_id)
            suites.append(
                WorkflowSuite(
                    suite_id=suite_id,
                    created_at=_parse_github_timestamp(
                        created_at,
                        context=f"CI workflow {run_id}",
                    ),
                    status=status,
                    conclusion=conclusion,
                )
            )
        fetched_count += len(page_runs)
        if fetched_count >= total_count:
            return tuple(suites)
        if not page_runs:
            raise SubmoduleCheckError(
                f"GitHub workflow pagination stopped early for {repository}@{revision}."
            )
        page += 1


def _parse_github_timestamp(value: str, *, context: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SubmoduleCheckError(
            f"GitHub returned an invalid timestamp for {context}."
        ) from error
    if parsed.tzinfo is None:
        raise SubmoduleCheckError(
            f"GitHub returned a timezone-free timestamp for {context}."
        )
    return parsed


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "yolo11-workspace-exact-sha-gate",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("SUBMODULE_CHECKS_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _read_github_json(
    request: urllib.request.Request,
    repository: str,
    revision: str,
) -> Any:
    """Read GitHub JSON with bounded retry for transient infrastructure errors."""

    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=20.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt == 2:
                raise SubmoduleCheckError(
                    f"Unable to read GitHub checks for {repository}@{revision}: {error}"
                ) from error
        except (OSError, urllib.error.URLError) as error:
            if attempt == 2:
                raise SubmoduleCheckError(
                    f"Unable to read GitHub checks for {repository}@{revision}: {error}"
                ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SubmoduleCheckError(
                f"GitHub returned unreadable check data for "
                f"{repository}@{revision}: {error}"
            ) from error
        time.sleep(0.5 * (2**attempt))
    raise AssertionError("GitHub retry loop exited without a result.")


def _check_run_order(
    run: Mapping[str, Any],
    check_name: str,
) -> tuple[tuple[datetime, int], int]:
    """Return a deterministic order and suite identity for a required check."""
    run_id = run.get("id")
    suite = run.get("check_suite")
    suite_id = suite.get("id") if isinstance(suite, Mapping) else None
    started_at = run.get("started_at")
    if (
        type(run_id) is not int
        or run_id <= 0
        or type(suite_id) is not int
        or suite_id <= 0
        or not isinstance(started_at, str)
    ):
        raise SubmoduleCheckError(
            f"GitHub returned incomplete identity data for required check "
            f"{check_name!r}."
        )
    started = _parse_github_timestamp(
        started_at,
        context=f"required check {check_name!r}",
    )
    return (started, run_id), suite_id


def _successful_evidence(
    check_runs: Sequence[Mapping[str, Any]],
    required_checks: Sequence[str],
    workflow_suites: Sequence[WorkflowSuite] | None = None,
) -> dict[str, str]:
    """Return successes from the latest suite that touched a required check.

    A commit can have simultaneous push, pull-request, and manually dispatched
    suites.  Suites are ordered by their authoritative creation time, then each
    name is resolved only within the selected suite.  This accepts one coherent
    push or pull-request suite even when downstream jobs finish out of order,
    but a newer partial, pending, or failed suite still blocks stale successes.
    """
    required_names = frozenset(required_checks)
    suites: dict[
        int,
        dict[str, tuple[tuple[datetime, int], Mapping[str, Any]]],
    ] = {}
    suite_created_at: dict[int, datetime] = {}
    suite_outcomes: dict[int, tuple[str, str | None]] = {}
    if workflow_suites is not None:
        for workflow_suite in workflow_suites:
            if workflow_suite.suite_id in suites:
                raise SubmoduleCheckError(
                    f"GitHub returned duplicate workflow suite "
                    f"{workflow_suite.suite_id}."
                )
            suites[workflow_suite.suite_id] = {}
            suite_created_at[workflow_suite.suite_id] = workflow_suite.created_at
            suite_outcomes[workflow_suite.suite_id] = (
                workflow_suite.status,
                workflow_suite.conclusion,
            )

    for run in check_runs:
        app = run.get("app")
        if not isinstance(app, Mapping) or app.get("slug") != "github-actions":
            continue
        name = run.get("name")
        if not isinstance(name, str) or name not in required_names:
            continue
        order, suite_id = _check_run_order(run, name)
        if workflow_suites is not None:
            if suite_id not in suites:
                continue
        else:
            suite = run.get("check_suite")
            raw_created_at = (
                suite.get("created_at") if isinstance(suite, Mapping) else None
            )
            if not isinstance(raw_created_at, str):
                raise SubmoduleCheckError(
                    f"GitHub returned incomplete suite identity data for required "
                    f"check {name!r}."
                )
            suite_created = _parse_github_timestamp(
                raw_created_at,
                context=f"check suite {suite_id}",
            )
            current_created = suite_created_at.setdefault(suite_id, suite_created)
            if current_created != suite_created:
                raise SubmoduleCheckError(
                    f"GitHub returned inconsistent creation times for check suite "
                    f"{suite_id}."
                )
        suite_runs = suites.setdefault(suite_id, {})
        current = suite_runs.get(name)
        if current is None or order > current[0]:
            suite_runs[name] = (order, run)

    if not suites:
        return {}

    latest_created_at = max(suite_created_at.values())
    frontier_suite_ids = tuple(
        suite_id
        for suite_id, created_at in suite_created_at.items()
        if created_at == latest_created_at
    )
    for suite_id in frontier_suite_ids:
        suite_runs = suites[suite_id]
        if (
            (
                workflow_suites is not None
                and suite_outcomes[suite_id] != ("completed", _SUCCESS)
            )
            or set(suite_runs) != required_names
            or any(
                run.get("status") != "completed" or run.get("conclusion") != _SUCCESS
                for _, run in suite_runs.values()
            )
        ):
            return {}
    latest_suite_id = max(frontier_suite_ids)

    evidence: dict[str, str] = {}
    for name, (_, run) in suites[latest_suite_id].items():
        if run.get("status") != "completed" or run.get("conclusion") != _SUCCESS:
            continue
        details_url = run.get("details_url")
        evidence[name] = details_url if isinstance(details_url, str) else ""
    return evidence


def verify_submodule_checks(
    root: Path,
    policy_path: Path,
    *,
    fetch_checks: CheckFetcher = _fetch_check_runs,
) -> dict[str, Any]:
    workspace_root = root.expanduser().resolve()
    verified: list[dict[str, Any]] = []
    for policy in _load_policies(workspace_root, policy_path):
        revision = _submodule_head_sha(workspace_root, policy.path)
        snapshot = fetch_checks(policy.repository, revision, policy.workflow)
        evidence = _successful_evidence(
            snapshot.runs,
            policy.required_checks,
            snapshot.workflow_suites,
        )
        missing = [name for name in policy.required_checks if name not in evidence]
        if missing:
            observed = sorted(
                str(run.get("name"))
                for run in snapshot.runs
                if isinstance(run.get("name"), str)
            )
            raise SubmoduleCheckError(
                f"Exact-SHA checks are not successful for "
                f"{policy.repository}@{revision}: missing={missing!r}; "
                f"observed={observed!r}"
            )
        verified.append(
            {
                "path": str(policy.path),
                "repository": policy.repository,
                "workflow": policy.workflow,
                "revision": revision,
                "checks": list(policy.required_checks),
            }
        )
    return {"event": "submodule_exact_sha_checks_verified", "projects": verified}


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parse_arguments(arguments)
    try:
        result = verify_submodule_checks(parsed.root, parsed.policy)
    except SubmoduleCheckError as error:
        print(
            json.dumps(
                {"event": "submodule_exact_sha_checks_failed", "error": str(error)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
