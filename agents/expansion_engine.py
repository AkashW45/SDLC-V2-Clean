"""
agents/expansion_engine.py  (FINAL)

ExpansionDecisionEngine — deterministic, policy-driven decision engine.

Takes:
  - an ASP (Adaptive Scope Profile),
  - a list of artifact candidates (generated artifacts with unit_counts / units_estimate / critic info),
  - optionally the number of already-accepted units,

and decides for each candidate:
  - accept              (auto-apply / queue side-effects),
  - queue_for_approval  (human gate),
  - reject              (policy or compliance violation).

Behavior:
  - respects forbidden_elements & mandatory_elements (hard blockers)
  - supports policy_mode: open | managed | conservative
  - uses unit_budgets (sum) and auto_approve_pct
  - computes / recomputes units deterministically with depth-aware unit_weights
  - sorts MVP-first, then by benefit_ratio (marginal_benefit / units)
  - caps runaway auto-expansion via max_consecutive_auto_expansions
  - logs decisions via audit()

Public API:
  decide(thread_id, asp, artifact_candidates, accepted_units_so_far=None, persist_audit=True) -> list[decision]
  decide_single(thread_id, asp, artifact, accepted_units_so_far=None) -> decision
  apply_decisions(artifacts, decisions) -> artifacts   (sets artifact['status'] + ['expansion_decision'])
"""

import math
import json
import logging
from typing import List, Dict, Any, Optional

try:
    from agents.stage2_store import load_unit_weights, audit
except Exception:  # standalone import / tests
    def load_unit_weights() -> Dict[str, Dict[int, float]]:
        return {}

    def audit(thread_id: str, actor: str, action: str, payload: Dict[str, Any]) -> None:
        logging.getLogger("ExpansionDecisionEngine").info(
            f"AUDIT(noop) thread={thread_id} actor={actor} action={action} payload={payload}"
        )


logger = logging.getLogger("ExpansionDecisionEngine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _text_of(obj: Any) -> str:
    """Serialize a JSON-like object to searchable lowercase text."""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False).lower()
    except Exception:
        try:
            return str(obj).lower()
        except Exception:
            return ""


def _compute_units_from_counts(
    unit_counts: Dict[str, float],
    unit_weights: Dict[str, Dict[int, float]],
    depth_level: int,
) -> int:
    """
    Compute deterministic units from unit_counts using a depth-aware unit_weights table.

      unit_counts:   {"FR": 3, "CODE_FILE": 2, "ENDPOINT": 5, ...}
      unit_weights:  {"FR": {1:2.0, 2:1.5, 3:1.2, ...}, "CODE_FILE": {...}, ...}
      depth_level:   1..5

    Returns ceil(sum). Unknown unit types fall back to weight 1.0.
    Missing depth keys snap to the nearest available depth.
    """
    if not unit_counts:
        return 0
    total = 0.0
    for k, count in unit_counts.items():
        if count is None:
            continue
        weights_for_k = unit_weights.get(k.upper(), {}) if unit_weights else {}
        weight = None
        if isinstance(weights_for_k, dict) and weights_for_k:
            weight = weights_for_k.get(depth_level)
            if weight is None:
                try:
                    nearest = min(weights_for_k.keys(), key=lambda d: abs(int(d) - depth_level))
                    weight = weights_for_k.get(nearest)
                except Exception:
                    weight = None
        if weight is None:
            weight = 1.0
        try:
            total += float(count) * float(weight)
        except Exception:
            total += 1.0
    return int(math.ceil(total))


def _estimate_units_from_artifact(
    artifact: Dict[str, Any],
    unit_weights: Dict[str, Dict[int, float]],
    depth_level: int,
) -> int:
    """
    Heuristic fallback if explicit unit_counts / units_estimate not present.
      - 'code' artifacts:  files length     -> CODE_FILE
      - 'brd' / 'prd':     FR count         -> FR
      - 'sprint':          jira_tickets     -> JIRA
      - 'arch':            nodes            -> NODE
      - generic dict:      count nested lists / dicts
    """
    if not artifact:
        return 0

    if isinstance(artifact.get("units_estimate"), (int, float)) and artifact.get("units_estimate") > 0:
        return int(math.ceil(artifact["units_estimate"]))

    unit_counts = artifact.get("unit_counts") or {}
    if unit_counts:
        return _compute_units_from_counts(unit_counts, unit_weights, depth_level)

    content = artifact.get("content") or artifact.get("body") or artifact
    typ = (artifact.get("artifact_type") or artifact.get("type") or "").lower()

    if typ in ("code", "patch") or (isinstance(content, dict) and "files" in content):
        files = artifact.get("files") or (content.get("files") if isinstance(content, dict) else []) or []
        return _compute_units_from_counts({"CODE_FILE": len(files)}, unit_weights, depth_level)

    if typ in ("brd", "prd") or (isinstance(content, dict) and "functional_requirements" in content):
        frs = content.get("functional_requirements") or [] if isinstance(content, dict) else []
        return _compute_units_from_counts({"FR": len(frs)}, unit_weights, depth_level)

    if typ in ("sprint", "sprint_plan") or (isinstance(content, dict) and "jira_tickets" in content):
        tickets = content.get("jira_tickets") or [] if isinstance(content, dict) else []
        return _compute_units_from_counts({"JIRA": len(tickets)}, unit_weights, depth_level)

    if typ in ("arch", "architecture") or (isinstance(content, dict) and "nodes" in content):
        nodes = content.get("nodes") or [] if isinstance(content, dict) else []
        return _compute_units_from_counts({"NODE": len(nodes)}, unit_weights, depth_level)

    if isinstance(content, dict):
        count = 0
        for v in content.values():
            if isinstance(v, list):
                count += len(v)
            elif isinstance(v, dict):
                count += 1
        if count > 0:
            return int(math.ceil(count))
    return 1


def _contains_forbidden(artifact: Dict[str, Any], forbidden_elements: List[str]) -> Optional[str]:
    """Return the matched forbidden element if any, else None. Normalizes _ - space variants."""
    if not forbidden_elements:
        return None
    text = _text_of(artifact)
    for f in forbidden_elements:
        if not f:
            continue
        variants = {f.lower(), f.lower().replace("_", "-"), f.lower().replace("_", " ")}
        for v in variants:
            if v in text:
                return f
    return None


def _missing_mandatory(artifact: Dict[str, Any], mandatory_elements: List[str]) -> List[str]:
    """Return list of missing mandatory elements (case-insensitive substring check)."""
    if not mandatory_elements:
        return []
    text = _text_of(artifact)
    missing = []
    for m in mandatory_elements:
        if not m:
            continue
        variants = {m.lower(), m.lower().replace("_", " ")}
        if not any(v in text for v in variants):
            missing.append(m)
    return missing


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def decide(
    thread_id: str,
    asp: Dict[str, Any],
    artifact_candidates: List[Dict[str, Any]],
    accepted_units_so_far: Optional[int] = None,
    persist_audit: bool = True,
) -> List[Dict[str, Any]]:
    """
    Decide accept / queue_for_approval / reject for each artifact candidate.

    artifact_candidates: each dict may include
      - artifact_id (str)
      - artifact_type / type (str)
      - category: 'mvp' | 'expansion'
      - confidence (0..100)
      - unit_counts (dict)        — preferred
      - units_estimate (int)      — overrides unit_counts if present
      - critic_verdict / critic_violations
      - marginal_benefit (float, 0..1)
      - evidence_resolved (bool)
      - content / body / files

    Returns: list[decision], one per candidate, in priority order.
    """
    unit_weights = load_unit_weights() or {}
    depth = int(asp.get("depth_level", 3))
    policy_mode = (asp.get("policy_mode") or "managed").lower()
    allow_unbounded = bool(asp.get("allow_unbounded", False))
    forbidden = asp.get("forbidden_elements", []) or []
    mandatory = asp.get("mandatory_elements", []) or []
    auto_pct = float(asp.get("auto_approve_pct", 0.0) or 0.0)
    expansion_policy = asp.get("expansion_policy", {}) or {}
    marginal_threshold = float(expansion_policy.get("marginal_benefit_threshold", 0.05))
    max_consecutive_auto = int(expansion_policy.get("max_consecutive_auto_expansions", 9999))

    # Budget: prefer unit_budgets dict; fall back to budget_estimates.work_units.estimate
    unit_budgets = asp.get("unit_budgets", {}) or {}
    if unit_budgets:
        try:
            budget_total = float(sum(float(v) for v in unit_budgets.values()))
        except Exception:
            budget_total = 0.0
    else:
        budget_total = float(
            asp.get("budget_estimates", {})
               .get("work_units", {})
               .get("estimate", 0) or 0
        )

    auto_limit = budget_total * auto_pct if auto_pct > 0 else 0.0

    # Determine already-accepted units
    if accepted_units_so_far is None:
        accepted_units_so_far = 0
        for a in artifact_candidates:
            already = (
                str(a.get("accepted", "")).lower() in ("true", "1")
                or str(a.get("status", "")).lower() == "accepted"
            )
            if already:
                accepted_units_so_far += _estimate_units_from_artifact(a, unit_weights, depth)

    logger.info(
        f"[ExpansionDecisionEngine] policy={policy_mode} unbounded={allow_unbounded} depth={depth} "
        f"budget_total={budget_total} auto_pct={auto_pct} auto_limit={auto_limit} "
        f"accepted_so_far={accepted_units_so_far} candidates={len(artifact_candidates)}"
    )

    # Enrich: recompute units + benefit ratio
    enriched: List[Dict[str, Any]] = []
    for a in artifact_candidates:
        if isinstance(a.get("units_estimate"), (int, float)) and a.get("units_estimate") >= 0:
            units = int(math.ceil(a["units_estimate"]))
        elif a.get("unit_counts"):
            units = _compute_units_from_counts(a["unit_counts"], unit_weights, depth)
        else:
            units = _estimate_units_from_artifact(a, unit_weights, depth)

        marginal_benefit = float(a.get("marginal_benefit", 0.0) or 0.0)
        benefit_ratio = (marginal_benefit / units) if units > 0 else marginal_benefit
        enriched.append({
            **a,
            "recomputed_units": int(units),
            "marginal_benefit": marginal_benefit,
            "benefit_ratio": benefit_ratio,
        })

    # Sort: MVP first, then benefit_ratio desc
    enriched.sort(
        key=lambda x: (1 if str(x.get("category", "")).lower() == "mvp" else 0,
                       x.get("benefit_ratio", 0.0)),
        reverse=True,
    )

    decisions: List[Dict[str, Any]] = []
    running_accepted = accepted_units_so_far
    consecutive_auto = 0

    for art in enriched:
        art_id = art.get("artifact_id") or art.get("id") or f"art-{art.get('artifact_type', 'unknown')}"
        art_type = art.get("artifact_type") or art.get("type") or "unknown"
        units = int(art.get("recomputed_units", 0))
        category = str(art.get("category", "")).lower()
        confidence = float(art.get("confidence", 0) or 0)
        critic_verdict = str(art.get("critic_verdict", "")).upper()
        critic_violations = art.get("critic_violations") or []
        evidence_resolved = art.get("evidence_resolved", None)
        is_load_bearing = art_type.lower() in ("code", "patch", "arch", "architecture")

        verdict = None
        reason = ""
        details: Dict[str, Any] = {
            "policy_mode": policy_mode,
            "depth_level": depth,
            "budget_total": budget_total,
            "auto_limit": auto_limit,
            "running_accepted_before": running_accepted,
            "marginal_benefit": art.get("marginal_benefit", 0.0),
            "benefit_ratio": art.get("benefit_ratio", 0.0),
            "confidence": confidence,
            "critic_verdict": critic_verdict,
        }

        # ---- 1. Hard reject: forbidden element
        forbidden_hit = _contains_forbidden(art, forbidden)
        if forbidden_hit:
            verdict = "reject"
            reason = f"Forbidden element detected: '{forbidden_hit}'"
            details["forbidden_match"] = forbidden_hit

        # ---- 2. Hard reject: critic says REGENERATE
        if verdict is None and critic_verdict == "REGENERATE":
            verdict = "reject"
            reason = "Critic verdict is REGENERATE — artifact must be regenerated before acceptance."
            details["critic_violations"] = critic_violations

        # ---- 3. Hard reject: critic flagged a critical violation
        if verdict is None:
            has_critical = any(
                str(v.get("severity", "")).lower() == "critical"
                for v in critic_violations if isinstance(v, dict)
            )
            if has_critical:
                verdict = "reject"
                reason = "Critic flagged a critical violation."
                details["critic_violations"] = critic_violations

        # ---- 4. Hard reject: load-bearing artifact with unresolved evidence
        if verdict is None and is_load_bearing and evidence_resolved is False:
            verdict = "reject"
            reason = (f"Evidence unresolved ({art.get('evidence_note', 'no evidence')}) — "
                      f"load-bearing artifact requires resolved evidence.")
            details["evidence_score"] = art.get("evidence_score", 0.0)

        # ---- 5. Queue: missing mandatory elements
        if verdict is None:
            missing = _missing_mandatory(art, mandatory)
            if missing:
                verdict = "queue_for_approval"
                reason = f"Missing mandatory elements: {missing}"
                details["missing_mandatory"] = missing

        # ---- 6. allow_unbounded → accept anything that passed hard blockers
        if verdict is None and allow_unbounded:
            verdict = "accept"
            reason = "allow_unbounded=true: auto-accept (no blockers)."

        # ---- 7. Conservative → never auto-accept
        if verdict is None and policy_mode == "conservative":
            verdict = "queue_for_approval"
            reason = "policy_mode=conservative requires human approval for all artifacts."

        # ---- 8. Open → accept unless something already blocked it
        if verdict is None and policy_mode == "open":
            verdict = "accept"
            reason = "policy_mode=open: auto-accept (no blockers)."

        # ---- 9. Managed → apply auto-accept criteria
        if verdict is None and policy_mode == "managed":
            critic_ok = (critic_verdict in ("", "ACCEPT")) and not critic_violations
            confidence_ok = confidence >= 75.0
            mvp_or_high_benefit = (category == "mvp") or (art.get("benefit_ratio", 0.0) >= marginal_threshold)
            within_auto_limit = (auto_limit <= 0) or ((running_accepted + units) <= auto_limit)
            within_total_budget = (budget_total <= 0) or ((running_accepted + units) <= budget_total)
            consec_ok = (category != "expansion") or (consecutive_auto < max_consecutive_auto)

            if (critic_ok and confidence_ok and mvp_or_high_benefit
                    and within_auto_limit and within_total_budget and consec_ok):
                verdict = "accept"
                reason = "managed mode: all auto-accept criteria met."
            else:
                verdict = "queue_for_approval"
                reasons = []
                if not critic_ok:
                    reasons.append("critic_not_clear")
                if not confidence_ok:
                    reasons.append(f"confidence<75 ({confidence})")
                if not mvp_or_high_benefit:
                    reasons.append(f"benefit_ratio<{marginal_threshold}")
                if not within_auto_limit:
                    reasons.append(f"would_exceed_auto_limit({auto_limit})")
                if not within_total_budget:
                    reasons.append(f"would_exceed_total_budget({budget_total})")
                if not consec_ok:
                    reasons.append("max_consecutive_auto_expansions_reached")
                reason = "managed mode: " + ", ".join(reasons)
                details["block_reasons"] = reasons

        # Fallback safety
        if verdict is None:
            verdict = "queue_for_approval"
            reason = "Unknown policy state — defaulting to human approval."

        # Acceptance bookkeeping
        if verdict == "accept":
            running_accepted += units
            if category == "expansion":
                consecutive_auto += 1
            else:
                consecutive_auto = 0
            details["running_accepted_after"] = running_accepted
        else:
            consecutive_auto = 0

        decision = {
            "artifact_id": art_id,
            "artifact_type": art_type,
            "verdict": verdict,
            "reason": reason,
            "recomputed_units": units,
            "details": details,
        }
        decisions.append(decision)

        if persist_audit:
            try:
                audit(thread_id, "ExpansionDecisionEngine", f"decision.{verdict}", {
                    "artifact_id": art_id,
                    "artifact_type": art_type,
                    "reason": reason,
                    "units": units,
                })
            except Exception as e:
                logger.warning(f"audit() failed for {art_id}: {e}")

        logger.info(f"[decision] {art_id} ({art_type}) -> {verdict} | {reason}")

    return decisions


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def decide_single(
    thread_id: str,
    asp: Dict[str, Any],
    artifact: Dict[str, Any],
    accepted_units_so_far: Optional[int] = None,
) -> Dict[str, Any]:
    """Decide for a single artifact."""
    results = decide(thread_id, asp, [artifact], accepted_units_so_far=accepted_units_so_far)
    return results[0] if results else {
        "artifact_id": artifact.get("artifact_id", "unknown"),
        "artifact_type": artifact.get("artifact_type", "unknown"),
        "verdict": "queue_for_approval",
        "reason": "no decision produced",
        "recomputed_units": 0,
        "details": {},
    }


def apply_decisions(artifacts: List[Dict[str, Any]],
                    decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Set artifact['status'] and artifact['expansion_decision'] from engine decisions.
    Matches on artifact_id. Returns the updated artifacts list.
    """
    status_map = {
        "accept": "accepted",
        "queue_for_approval": "queued",
        "reject": "rejected",
    }
    by_id = {d["artifact_id"]: d for d in decisions}
    for art in artifacts:
        aid = art.get("artifact_id") or art.get("id")
        d = by_id.get(aid)
        if d:
            art["status"] = status_map.get(d["verdict"], "candidate")
            art["expansion_decision"] = d
            art["units_estimate"] = d["recomputed_units"]
    return artifacts
