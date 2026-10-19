# YOLO11 Workspace

This directory is the integration boundary for the production inference and
training applications. The two child directories remain independent Git
repositories; `workspace.yaml` is the single source of truth for cross-project
paths.

```text
yolo11_workspace/
├── .gitmodules
├── workspace.yaml
├── Yolo11_auto_train/
└── yolo11_inference/
```

- Use `start_inference.bat` for production inference.
- Use `start_training.bat` for the training GUI or pass an operator handoff JSON
  path as its first argument.
- Do not rename either child directory without updating `workspace.yaml`.
- Runtime models remain owned by `yolo11_inference/models`; reviewed datasets,
  operator jobs, and training runs remain owned by `Yolo11_auto_train`.
- Inspection images and the inspection-history database are stored in the
  workspace-level `Result` directory configured by `paths.inference_results`.

## Architecture

Read these in order:

1. [`Yolo11_auto_train/docs/ARCHITECTURE.md`](Yolo11_auto_train/docs/ARCHITECTURE.md)
   §1: every file contract between the two projects (handoff JSON, station model
   directory, acceptance gate, shared locks, color fixture). It also covers the
   training pipeline, the deploy gates and autotrain.
2. [`yolo11_inference/docs/architecture/MODULE_ARCHITECTURE.md`](yolo11_inference/docs/architecture/MODULE_ARCHITECTURE.md)
   ([繁中](yolo11_inference/docs/architecture/MODULE_ARCHITECTURE_ZH_TW.md)): the
   production runtime, its subsystem map and its concurrency model.
3. [`yolo11_inference/docs/DOCUMENTATION_INDEX.md`](yolo11_inference/docs/DOCUMENTATION_INDEX.md):
   role-based entry point to the operations, data and model-lifecycle docs.

The loop the two projects form: inspection → `Result/` → operator review →
handoff to training → retrain → gates → `deploy` publishes into
`yolo11_inference/models` → inspection. The training project's `deploy` task is
the only code path that writes production models.

## Clone

Clone the workspace and both child repositories together:

```powershell
git clone --recurse-submodules https://github.com/DioWang67/Yolo11_workspace.git
```

For an existing clone:

```powershell
git submodule update --init --recursive
```

## Development workflow

Commit and push changes inside the owning child repository first. Then update
the workspace pointer in a separate commit:

```powershell
git add Yolo11_auto_train yolo11_inference
git commit -m "Update workspace project revisions"
```

This repository intentionally does not copy child repository history, model
weights, datasets, inspection results, or local runtime logs.

Select a local Python interpreter in VS Code after cloning. The shared workspace
does not pin an absolute Conda or Python installation path because developer and
station layouts may differ.

## Continuous integration

The workspace workflow checks out the exact submodule revisions, rejects
uninitialized or mismatched submodules, requires successful named GitHub checks
for those exact child commits, reruns both projects' blocking lint and type-check
gates, validates `workspace.yaml` through both path implementations, and
publishes JUnit reports. Each child repository still owns its full Linux/Windows
test, coverage, runtime, executable, and package gates.

CI actions are pinned to immutable commit SHAs. Dependabot groups the scheduled
GitHub Actions, Python dependency, and submodule updates so those pins remain
maintainable.

Configure the optional `SUBMODULE_CHECKS_TOKEN` Actions secret with read-only
Checks access to both child repositories to raise the GitHub API rate limit.
Public repositories also work without it, using GitHub's unauthenticated limit.

For an inference gitlink on a feature branch, the normal push and pull-request
runs skip `Verified Windows executable`. Dispatch the child `CI` workflow on the
feature branch whose HEAD is the exact gitlink SHA before updating this
workspace:

```powershell
gh workflow run ci.yml --repo DioWang67/Yolo_anomalib --ref <feature-branch>
```

The workspace gate remains closed until one current child check suite has every
required check, including the Windows executable, completed successfully.

## Merge gate

Before updating `main`, require successful child-repository checks for the exact
two gitlink commits, successful Workspace CI, completed operational acceptance
for production-facing changes, and substantive review of each child PR. If a
child PR is squash- or rebase-merged, update the workspace gitlink to the final
commit reachable from that child repository's default branch and rerun all
workspace checks.

Duplicate-filter code may be integrated before the 500-cycle/full-shift pilot
only while the production configuration remains `report_only` (or disabled).
Changing it to `suppress` is a production-facing activation and still requires
the documented pilot evidence and named approvals.
