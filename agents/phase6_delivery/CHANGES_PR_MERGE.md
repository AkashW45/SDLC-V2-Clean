# PR Merge Integration (Phase 6 → Phase 7)

## What changed

Before this change, the pipeline had a critical gap: Phase 6 opened a PR on a
feature branch, the human clicked "approve" on the dashboard, and Phase 7
then cloned `main` and started building. But **nothing actually merged the
PR**, so Phase 7 deployed whatever was on `main` before Phase 6 ran — not
the new code.

This change closes that gap. After the human approves Phase 6, the API now:

1. Merges each PR via GitHub's API (using `merge_method=squash`)
2. Records the resulting commit SHA per repo
3. Refuses to start Phase 7 if any merge failed
4. Pins Phase 7's `git checkout` to those exact SHAs

So the code that gets deployed is **exactly** the code that was reviewed —
reproducible, audit-trail-friendly, and immune to other developers landing
unrelated changes on `main` in between.

## Files modified

| File | Change |
|---|---|
| `agents/phase6_delivery/delivery_agent.py` | Added `_parse_pr_url`, `_merge_one_pr`, `merge_prs_for_state`, `merge_prs` (graph node); wired `merge_prs` into the delivery graph after `process_approval`; expanded `DeliveryState` with `merged_shas` and `merge_errors` |
| `agents/phase7_deployment/deployment_agent.py` | Added `merged_shas` to `DeploymentState`; `resolve_deploy_sequence` now sets `rc.ref` from `merged_shas` so the executor's `git checkout` lands on the exact merged commit |
| `api/main.py` | Approval handler for Phase 6→7 transition now runs `merge_prs_for_state` and persists `merged_shas`/`merge_errors`/`phase6_final_status` into pipeline state; `run_phase7` refuses to run on `MERGE_FAILED`; `_safe_state` allowlist extended to persist the new fields |

## End-to-end flow

```
User clicks Approve on Phase 6 in dashboard
        │
        ▼
POST /pipeline/{id}/approve  (api/main.py)
        │
        ├── audit "APPROVED"
        ├── set status = PHASE_6_APPROVED
        │
        ▼
   next_phase == "7"  →  RUN MERGE STEP
        │
        ▼
   merge_prs_for_state(state)
        │
        ├── PR_REJECTED?            → skip merge, keep status
        ├── PR_SKIPPED_NEW_PROJECT? → no PR, status=READY_FOR_DEPLOYMENT
        ├── no pr_urls?             → nothing to do, status=READY_FOR_DEPLOYMENT
        │
        ▼  (normal path)
   for each pr_url:
        ├── parse owner/repo/pr_number
        ├── PUT /repos/{owner}/{repo}/pulls/{n}/merge  (squash)
        ├── on 200: record merged_shas[repo] = response.sha
        └── on error: append to merge_errors
        │
        ├── any failures?  → status=MERGE_FAILED   (Phase 7 will refuse)
        └── all succeeded? → status=READY_FOR_DEPLOYMENT
        │
        ▼
Persist merged_shas + phase6_final_status into pipeline state
        │
        ▼
   background_tasks.add_task(run_phase7, ...)
        │
        ▼
   run_phase7 checks phase6_final_status:
        ├── MERGE_FAILED         → set ERROR, save, return
        └── READY_FOR_DEPLOYMENT → proceed
        │
        ▼
   build_deployment_graph().invoke(DeploymentState(merged_shas=...))
        │
        ▼
   resolve_deploy_sequence:
        └── for each repo: rc.ref = merged_shas[name]
        │
        ▼
   fetch_repos (executor):
        └── git checkout <merged_sha>   (NOT git pull)
        │
        ▼
   build_images, push to ECR, deploy to ECS
```

## What happens when a merge fails

GitHub's PR-merge endpoint returns these failures most often:

| HTTP | Cause | What the user does |
|---|---|---|
| 405 | Conflicts with main, or required CI checks haven't passed, or branch protection requires reviews | Resolve on GitHub UI, then `POST /pipeline/{id}/resume {"phase": 7}` |
| 409 | Head SHA changed while we were trying to merge (someone else pushed) | Same — retry triggers a fresh merge attempt |
| 422 | PR is closed/draft/already merged | Investigate; usually means the user merged it manually |

The pipeline lands in status `MERGE_FAILED` with the specific GitHub error
in `merge_errors`. Phase 7 won't auto-retry — the user must fix the PR and
explicitly resume. This is intentional: silent retries can mask conflicts.

## Why merge_method = "squash"

The generated PR usually contains many small commits (one per file in the
case of the diff-based generator). Squashing them into a single commit on
main means:

- One clean commit per requirement, matching one Jira ticket
- Easy `git log` / blame on main
- `git revert` is one operation, not N

If you'd prefer `merge` (preserve all commits) or `rebase` (replay each onto
main), change the `payload["merge_method"]` in `_merge_one_pr` to one of
those strings.

## Why a separate `merge_prs_for_state` helper

The same merge logic needs to be callable from two places:

1. The LangGraph delivery graph (as the `merge_prs` node, after
   `process_approval`)
2. The API endpoint (`api/main.py` approval handler), which currently
   bypasses Phase 6's graph entirely and just spawns Phase 7 as a
   background task

Rather than implement merge twice, the logic lives in
`merge_prs_for_state` and both callers delegate to it. Single source of
truth.

## Why NOT in `pr_manager.py`

`agents/pr_manager.py` has a module-level docstring that explicitly says
"Never auto-merges. Never force-pushes." That's a deliberate invariant for
PR *creation* — PR creation should be cheap and reversible. Merging happens
only AFTER human approval, so it belongs in the delivery agent, not the PR
manager. The two responsibilities are kept clean: pr_manager creates,
delivery_agent merges-after-approval.

## Required environment

The merge call needs the same GitHub credentials Phase 6 already uses:

```
GITHUB_TOKEN=<personal access token or app token with `repo:public_repo`
              or `repo` for private repos>
GITHUB_REPO_OWNER=AkashW45
```

If branch protection on the target repo requires:

- **Required reviews** — your GitHub App or token needs review/admin power
  on the repo, OR you must lower the required-reviews threshold for
  AI-generated PRs.
- **Required status checks** — wait for CI before approving, so the checks
  have time to pass. The merge endpoint will return 405 if a required
  check is still pending.

If you want a permissive setup for development: in the target repo, go
to Settings → Branches → main → "Allow administrators to bypass" and
ensure the SDLC bot is an admin.

## Testing without a real merge

For dev work, set `dry_run: true` in `config/deployment.yaml` to make
Phase 7 a no-op. The merge step itself, however, always talks to GitHub
for real — if you need to mock it, point `GITHUB_TOKEN` at a fake server
or temporarily set the merge URL to a localhost stub.
