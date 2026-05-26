"""
ALB provisioning + detection for Phase 7 ECS deployments.

Goal: give an ECS (Fargate) service a STABLE, demoable URL by putting an
Application Load Balancer in front of it — but be SMART about it:

  * GREENFIELD  (service does not exist yet)
        -> create ALB + target group + listener + ALB security group,
           tag them with sdlc:service=<name>, and hand the target group ARN
           back so the service is created already wired to the ALB.
           Return the ALB DNS name as the endpoint.

  * BROWNFIELD-WITH-ALB  (service exists AND is already attached to a LB)
        -> DO NOT create anything. Read the existing wiring straight off the
           service (describe-services -> loadBalancers[].targetGroupArn),
           follow it to the load balancer, and reuse its DNS name as-is.
           Caller just does update-service for the new image.

  * BROWNFIELD-LB-LESS  (service exists but has NO load balancer)
        -> DO NOT create an ALB. An ECS service created without a load
           balancer CANNOT have one attached via update-service (AWS only
           accepts the loadBalancers block at create-service time). So we
           leave its networking untouched, update the image only, and log
           that attaching an ALB requires a deliberate one-time recreate.

DETECTION is keyed off the ECS service's own `loadBalancers` array — the
structural link AWS itself maintains — NOT off any ALB naming convention.
That means a brownfield app whose ALB was created by some other tool, with a
completely different name, is still detected correctly and never duplicated.

Every call honours config.dry_run and returns structured dicts so the trace
is never silent.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# NOTE: _run / _ok / _fail / _aws_env / _aws_region_args / _ecs_describe_json /
# _opt live in deployment_executor. To avoid a circular import at module load
# time we import them lazily inside the functions that need them.


# ---------------------------------------------------------------------------
# Detection — the keystone. Answer: "is THIS service already behind an ALB?"
# ---------------------------------------------------------------------------

def detect_service_alb(cluster: str, service: str,
                       env: Dict[str, str], region_args: List[str],
                       *, dry_run: bool) -> Dict[str, Any]:
    """Classify a service into greenfield / brownfield-with-alb / brownfield-lb-less.

    Returns a dict:
      {
        "state": "GREENFIELD" | "BROWNFIELD_WITH_ALB" | "BROWNFIELD_LB_LESS",
        "service_exists": bool,
        "target_group_arn": <arn or None>,   # only for BROWNFIELD_WITH_ALB
        "container_name": <str or None>,      # from the existing LB wiring
        "container_port": <int or None>,
      }

    Detection is purely from `describe-services` -> services[0].loadBalancers,
    which is the relationship AWS maintains regardless of how anything is named.
    """
    from core.deployment_executor import _ecs_describe_json

    if dry_run:
        # In dry-run we can't know real AWS state; assume greenfield so the
        # log shows the full create path without touching anything.
        print("  [alb-detect] dry-run — assuming GREENFIELD (no AWS calls)")
        return {"state": "GREENFIELD", "service_exists": False,
                "target_group_arn": None, "container_name": None,
                "container_port": None}

    svc_json = _ecs_describe_json(
        ["aws", "ecs", "describe-services", "--cluster", cluster,
         "--services", service, *region_args],
        env=env, dry_run=False)

    services = (svc_json or {}).get("services", [])
    active = [s for s in services if s.get("status") == "ACTIVE"]
    if not active:
        print(f"  [alb-detect] service '{service}' not found -> GREENFIELD")
        return {"state": "GREENFIELD", "service_exists": False,
                "target_group_arn": None, "container_name": None,
                "container_port": None}

    svc = active[0]
    load_balancers = svc.get("loadBalancers", []) or []
    if load_balancers:
        lb = load_balancers[0]
        tg_arn = lb.get("targetGroupArn")
        print(f"  [alb-detect] service '{service}' already attached to a load "
              f"balancer (target group: {tg_arn}) -> BROWNFIELD_WITH_ALB")
        return {
            "state": "BROWNFIELD_WITH_ALB",
            "service_exists": True,
            "target_group_arn": tg_arn,
            "container_name": lb.get("containerName"),
            "container_port": lb.get("containerPort"),
        }

    print(f"  [alb-detect] service '{service}' exists but has NO load balancer "
          f"-> BROWNFIELD_LB_LESS")
    return {"state": "BROWNFIELD_LB_LESS", "service_exists": True,
            "target_group_arn": None, "container_name": None,
            "container_port": None}


def dns_from_target_group(tg_arn: str, env: Dict[str, str],
                          region_args: List[str],
                          *, dry_run: bool) -> Optional[str]:
    """Given a target group ARN, walk to its load balancer and return the
    DNS name (http://<dns>). Used for the BROWNFIELD_WITH_ALB reuse path."""
    from core.deployment_executor import _ecs_describe_json

    if dry_run or not tg_arn:
        return None

    tg_json = _ecs_describe_json(
        ["aws", "elbv2", "describe-target-groups",
         "--target-group-arns", tg_arn, *region_args],
        env=env, dry_run=False)
    tgs = (tg_json or {}).get("TargetGroups", [])
    if not tgs:
        return None
    lb_arns = tgs[0].get("LoadBalancerArns", [])
    if not lb_arns:
        return None

    lb_json = _ecs_describe_json(
        ["aws", "elbv2", "describe-load-balancers",
         "--load-balancer-arns", lb_arns[0], *region_args],
        env=env, dry_run=False)
    lbs = (lb_json or {}).get("LoadBalancers", [])
    if not lbs:
        return None
    dns = lbs[0].get("DNSName")
    return f"http://{dns}" if dns else None


# ---------------------------------------------------------------------------
# Greenfield provisioning — create the ALB stack and return the TG ARN so the
# service can be created already wired to it.
# ---------------------------------------------------------------------------

def ensure_alb_stack(service: str, vpc_id: str, subnets: List[str],
                     container_port: int, health_check_path: str,
                     env: Dict[str, str], region_args: List[str],
                     *, dry_run: bool,
                     hc_interval: int = 10, hc_healthy_threshold: int = 2,
                     hc_timeout: int = 5,
                     deregistration_delay: int = 30) -> Dict[str, Any]:
    """Create (idempotently) an ALB + target group + listener + ALB SG for a
    greenfield service. Every resource is tagged sdlc:service=<service> so our
    OWN future runs can also discover it by tag if ever needed.

    Health-check tuning (the main bring-up latency lever):
      hc_interval (default 10s)          — how often the ALB probes the target.
      hc_healthy_threshold (default 2)   — consecutive passes before routing.
      hc_timeout (default 5s)            — per-probe timeout.
      deregistration_delay (default 30s) — connection-drain on redeploy (the
                                           AWS default is 300s; 30s makes
                                           redeploys roll far faster).
    AWS target-group DEFAULTS are 30s interval + 5 healthy threshold = up to
    ~150s before a fresh target serves traffic. The defaults here bring that
    to ~20s (2 × 10s) while staying safely above an app's startup jitter. All
    are configurable from the deploy target / .deploy.yaml — nothing is
    hard-coded away.

    Returns:
      {"ok": True, "target_group_arn": <arn>, "alb_dns": "http://...",
       "alb_sg_id": <sg>, "endpoint": "http://..."}
    or a _fail dict.

    Requires >=2 subnets in different AZs (ALB constraint). We pass up to 3
    and let ELBv2 validate AZ spread.
    """
    from core.deployment_executor import _run, _ecs_describe_json, _fail

    alb_name = f"sdlc-{service}-alb"[:32]        # ELBv2 name limit = 32 chars
    tg_name = f"sdlc-{service}-tg"[:32]
    sg_name = f"sdlc-{service}-alb-sg"
    tag_args = ["--tags", f"Key=sdlc:service,Value={service}",
                "Key=sdlc:managed,Value=true"]

    if dry_run:
        print(f"  [alb-provision] dry-run — would create ALB '{alb_name}', "
              f"target group '{tg_name}', listener :80 -> :{container_port}, "
              f"health check '{health_check_path}'")
        return {"ok": True,
                "target_group_arn": f"arn:aws:dry-run:targetgroup/{tg_name}",
                "alb_dns": f"http://{alb_name}.dry-run.elb.amazonaws.com",
                "alb_sg_id": "sg-dryrun",
                "endpoint": f"http://{alb_name}.dry-run.elb.amazonaws.com"}

    if len(subnets) < 2:
        return _fail("ensure_alb_stack",
                     f"ALB needs >=2 subnets in different AZs; got {len(subnets)}. "
                     f"Use a VPC with multiple subnets or set subnets in config.",
                     service=service)

    # 1) ALB security group — opens 80 to the world.
    alb_sg_id = ""
    sg_json = _ecs_describe_json(
        ["aws", "ec2", "describe-security-groups", "--filters",
         f"Name=group-name,Values={sg_name}",
         f"Name=vpc-id,Values={vpc_id}", *region_args],
        env=env, dry_run=False)
    if sg_json and sg_json.get("SecurityGroups"):
        alb_sg_id = sg_json["SecurityGroups"][0]["GroupId"]
    else:
        created = _ecs_describe_json(
            ["aws", "ec2", "create-security-group", "--group-name", sg_name,
             "--description", f"SDLC ALB SG for {service}",
             "--vpc-id", vpc_id, *region_args],
            env=env, dry_run=False)
        alb_sg_id = (created or {}).get("GroupId", "")
        if alb_sg_id:
            _run(["aws", "ec2", "authorize-security-group-ingress",
                  "--group-id", alb_sg_id, "--protocol", "tcp",
                  "--port", "80", "--cidr", "0.0.0.0/0", *region_args],
                 env=env, dry_run=False, check=False, timeout=60)
    if not alb_sg_id:
        return _fail("ensure_alb_stack", "could not resolve/create ALB SG",
                     service=service)

    # 2) Load balancer — reuse if a same-named one exists, else create.
    lb_arn = ""
    lb_dns = ""
    existing_lb = _ecs_describe_json(
        ["aws", "elbv2", "describe-load-balancers", "--names", alb_name,
         *region_args], env=env, dry_run=False)
    if existing_lb and existing_lb.get("LoadBalancers"):
        lb_arn = existing_lb["LoadBalancers"][0]["LoadBalancerArn"]
        lb_dns = existing_lb["LoadBalancers"][0]["DNSName"]
    else:
        create_lb = _ecs_describe_json(
            ["aws", "elbv2", "create-load-balancer", "--name", alb_name,
             "--type", "application", "--scheme", "internet-facing",
             "--subnets", *subnets,
             "--security-groups", alb_sg_id,
             *tag_args, *region_args],
            env=env, dry_run=False)
        lbs = (create_lb or {}).get("LoadBalancers", [])
        if not lbs:
            return _fail("ensure_alb_stack",
                         "create-load-balancer returned no LoadBalancer "
                         "(check subnet AZ spread / permissions)",
                         service=service)
        lb_arn = lbs[0]["LoadBalancerArn"]
        lb_dns = lbs[0]["DNSName"]

    # 3) Target group — target-type ip is REQUIRED for Fargate awsvpc tasks.
    #    Health-check timings are tuned for fast bring-up (see signature docs):
    #    a fresh target serves traffic after ~hc_healthy_threshold × hc_interval
    #    instead of the AWS default ~150s.
    tg_arn = ""
    existing_tg = _ecs_describe_json(
        ["aws", "elbv2", "describe-target-groups", "--names", tg_name,
         *region_args], env=env, dry_run=False)
    if existing_tg and existing_tg.get("TargetGroups"):
        tg_arn = existing_tg["TargetGroups"][0]["TargetGroupArn"]
    else:
        create_tg = _ecs_describe_json(
            ["aws", "elbv2", "create-target-group", "--name", tg_name,
             "--protocol", "HTTP", "--port", str(container_port),
             "--vpc-id", vpc_id, "--target-type", "ip",
             "--health-check-path", health_check_path,
             "--health-check-protocol", "HTTP",
             "--health-check-interval-seconds", str(hc_interval),
             "--healthy-threshold-count", str(hc_healthy_threshold),
             "--health-check-timeout-seconds", str(hc_timeout),
             *tag_args, *region_args],
            env=env, dry_run=False)
        tgs = (create_tg or {}).get("TargetGroups", [])
        if not tgs:
            return _fail("ensure_alb_stack",
                         "create-target-group returned no TargetGroup",
                         service=service)
        tg_arn = tgs[0]["TargetGroupArn"]

    # 3b) Shorten the connection-drain (deregistration) delay from the AWS
    #     default of 300s. This does NOT affect first bring-up, but makes
    #     REDEPLOYS roll the old task out quickly instead of lingering 5 min.
    _run(["aws", "elbv2", "modify-target-group-attributes",
          "--target-group-arn", tg_arn,
          "--attributes",
          f"Key=deregistration_delay.timeout_seconds,Value={deregistration_delay}",
          *region_args],
         env=env, dry_run=False, check=False, timeout=30)

    # 4) Listener :80 -> forward to target group. Idempotent: skip if a
    #    listener on :80 already exists for this LB.
    listeners = _ecs_describe_json(
        ["aws", "elbv2", "describe-listeners",
         "--load-balancer-arn", lb_arn, *region_args],
        env=env, dry_run=False)
    has_80 = any(l.get("Port") == 80
                 for l in (listeners or {}).get("Listeners", []))
    if not has_80:
        _run(["aws", "elbv2", "create-listener",
              "--load-balancer-arn", lb_arn,
              "--protocol", "HTTP", "--port", "80",
              "--default-actions",
              f"Type=forward,TargetGroupArn={tg_arn}",
              *region_args],
             env=env, dry_run=False, check=False, timeout=60)

    endpoint = f"http://{lb_dns}" if lb_dns else None
    print(f"  [alb-provision] ALB ready for '{service}': {endpoint}")
    return {"ok": True, "target_group_arn": tg_arn, "alb_dns": endpoint,
            "alb_sg_id": alb_sg_id, "endpoint": endpoint}


def wait_targets_healthy(tg_arn: str, env: Dict[str, str],
                         region_args: List[str], *, dry_run: bool,
                         timeout: int = 300,
                         poll_interval: int = 3) -> Dict[str, Any]:
    """Poll target health until at least one target is 'healthy'. An ALB
    serves 503 until a target passes its health check, so this prevents a
    smoke test from racing the load balancer.

    poll_interval (default 3s) controls how often we check — fine-grained so
    we return within a few seconds of the target actually going healthy,
    rather than overshooting. This is pure observation; it does not change
    when AWS marks the target healthy, only how quickly we notice.

    Best-effort: a timeout here is reported but NOT treated as a hard deploy
    failure (the deploy itself may still be settling)."""
    from core.deployment_executor import _run, _ecs_describe_json

    if dry_run or not tg_arn:
        return {"healthy": True, "dry_run": dry_run}

    import time
    deadline = time.time() + timeout
    last_states: List[str] = []
    while time.time() < deadline:
        health = _ecs_describe_json(
            ["aws", "elbv2", "describe-target-health",
             "--target-group-arn", tg_arn, *region_args],
            env=env, dry_run=False)
        descs = (health or {}).get("TargetHealthDescriptions", [])
        last_states = [d.get("TargetHealth", {}).get("State", "?") for d in descs]
        if any(s == "healthy" for s in last_states):
            print(f"  [alb-health] target(s) healthy: {last_states}")
            return {"healthy": True, "states": last_states}
        print(f"  [alb-health] waiting for healthy target — current: "
              f"{last_states or 'none registered yet'}")
        time.sleep(poll_interval)

    print(f"  [alb-health] ⚠ timed out after {timeout}s — last states: "
          f"{last_states}. Deploy may still be settling; check the ALB.")
    return {"healthy": False, "states": last_states, "timed_out": True}
