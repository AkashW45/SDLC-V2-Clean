# CI/CD Generation — Brownfield Fix

## The bug

The first version of `generate_cicd_node` had two stacked bugs that made it
overwrite hand-tuned infra in brownfield repos:

1. **`existing_files_in_repo` was sourced from `state.existing_code`**, which
   only contains files Phase 3 flagged as "affected" — typically narrow
   slice like `app/models.py` and `app/routes.py`. Infrastructure files
   like `Dockerfile`, `.deploy.yaml`, and `.github/workflows/ci-cd.yml`
   are never in that list because they're not relevant to the requirement.

2. **`_detect_existing` then asked "is `Dockerfile` in this set?"** — got
   `False` even when the file existed in the repo on disk — and told the
   LLM "you need to generate one." The LLM dutifully said yes, and we
   ended up generating a second Dockerfile on top of the existing one.

Net effect: every brownfield run clobbered existing deployment infra.

## The fix

Two changes:

### 1. Look at what's actually on disk, not at the impact-report subset

`generate_cicd_node` now:

- Determines `is_brownfield` from `impact_report.affected_files` being
  non-empty AND state not being `NEW_PROJECT_NO_CODE`.
- For brownfield runs, scans each cloned repo's root directory + the
  `.github/workflows/` subdirectory directly (via `os.listdir`). This
  reveals the **actual** set of files present, not just the ones Phase 3
  cared about.
- Passes the real set to `generate_cicd_artifacts` along with a new
  `is_brownfield` flag.

### 2. Apply a brownfield-specific policy

`generate_cicd_artifacts` now branches on `is_brownfield`:

| Case | What we do |
|---|---|
| Brownfield with full infra (all three present) | Generate nothing. Log "respecting existing infra." |
| Brownfield with partial infra (some present, some missing) | Generate nothing. Warn the user. Rationale: mixing AI-generated infra with hand-tuned infra is more likely to break things than fix them — the human knows their existing setup, the AI doesn't. |
| Brownfield with no infra at all (legacy service deployed manually) | Fall through to LLM-driven decision (same path as greenfield), but log a loud warning telling the user to review carefully. |
| Greenfield | Generate the full stack as before. |

The "partial infra → do nothing" choice deserves explanation. It's tempting
to say "well, we have a Dockerfile, just generate the missing `.deploy.yaml`."
But the team's existing Dockerfile encodes how they build the image — base
image, build steps, CMD. A `.deploy.yaml` generated without knowing those
specifics will reference the wrong port, the wrong service name, or the
wrong registry. Better to refuse and tell the user.

## Files changed in this fix

| File | Change |
|---|---|
| `agents/phase4_codegen/codegen_agent.py` | `generate_cicd_node` now scans the cloned repo on disk for actual existing files, detects brownfield vs greenfield from state, passes both into the artifact generator. |
| `agents/phase4_codegen/cicd_generator.py` | Added `is_brownfield` parameter to `generate_cicd_artifacts`; added `_brownfield_decision` that respects existing infra and only falls through to the LLM when no infra exists at all; `.dockerignore` only generated if missing. |

## Concrete examples

### Example A — brownfield with full infra (the most common case)

```
leave-mgmt-backend/
  Dockerfile                         ← exists
  .deploy.yaml                       ← exists
  .github/workflows/ci-cd.yml        ← exists
  app/models.py                      ← Phase 4 modified this
```

Run output:
```
[cicd] brownfield infra detected on disk: ['.deploy.yaml', '.github/workflows/ci-cd.yml', 'Dockerfile']
[cicd] brownfield repo has full infra — nothing to generate
[cicd] generated 0 CI/CD artifact(s)
```

PR contains only `app/models.py` changes — no clobbered infra.

### Example B — brownfield with partial infra

```
leave-mgmt-backend/
  Dockerfile                         ← exists (hand-tuned)
  (no .deploy.yaml)
  (no workflow)
  app/models.py                      ← Phase 4 modified this
```

Run output:
```
[cicd] brownfield infra detected on disk: ['Dockerfile']
[cicd] ⚠ brownfield repo has partial infra. Present: Dockerfile=True,
       .deploy.yaml=False, workflow=False. Missing: ['.deploy.yaml',
       '.github/workflows/*.yml']. NOT auto-generating — please add by
       hand to ensure consistency with your existing infra.
[cicd] generated 0 CI/CD artifact(s)
```

The user gets a clear message that something is missing and needs to be
addressed manually.

### Example C — brownfield with no infra (legacy service)

```
leave-mgmt-backend/                 ← deployed via "scp + supervisorctl"
  (no Dockerfile)
  (no .deploy.yaml)
  (no workflow)
  app/models.py                      ← Phase 4 modified this
```

Run output:
```
[cicd] brownfield repo has NO infra files — falling through to LLM-driven
       generation. Review carefully against your existing deployment
       process before merging.
[cicd] generated 4 CI/CD artifact(s)
       + Dockerfile
       + .dockerignore
       + .deploy.yaml
       + .github/workflows/ci-cd.yml
```

User sees the warning, reviews the PR, decides whether to migrate from
manual deploys to Docker/ECS.

### Example D — greenfield (no change in behavior)

```
(empty repo, just created)
```

Run output:
```
[cicd] greenfield project — generating full deployment stack
[cicd] generated 4 CI/CD artifact(s)
```

## How to verify on your end

After applying this fix, run a pipeline on a brownfield repo that already
has a Dockerfile. Check Phase 4's output — `generated_changes` should NOT
include `Dockerfile`. Look for either of these log lines:

- `brownfield repo has full infra — nothing to generate`
- `brownfield repo has partial infra. ... NOT auto-generating`

If you still see `+ Dockerfile` in the output for a repo that already has
one, the on-disk scan isn't finding the file. Check that:

- `ensure_repo_cloned(repo_name)` succeeded (look for clone-failure
  messages in the Phase 4 log)
- `get_repo_local_path(repo_name)` resolves to the directory you expect
- The Dockerfile is at the repo root, not in a subdirectory (the scanner
  is intentionally shallow — it only looks at the top level + the one
  known infra subdirectory `.github/workflows/`)
