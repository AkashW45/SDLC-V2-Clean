"""
Phase 6 — Delivery Agent
Pushes generated code changes to GitHub and creates PRs.
Human approval gate — engineers must review PR before merge.
"""

import os
import json
import subprocess
import tempfile
import shutil
import datetime
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_REPO_OWNER")


# -----------------------------------------
# State
# -----------------------------------------

class DeliveryState(TypedDict):
    requirement: str
    generated_changes: list
    test_files: list
    branch_name: str
    repo_url: str
    pr_urls: list
    human_feedback: str
    approved: bool
    status: str


# -----------------------------------------
# Git Helpers
# -----------------------------------------

def run_git(cmd: list, cwd: str) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Git error: {result.stderr.strip()}")
    return result.stdout.strip()


def push_files_to_branch(
    repo_url: str,
    branch_name: str,
    files: list,
    commit_message: str
) -> dict:
    """Clone repo, write files, push to branch."""
    temp_dir = tempfile.mkdtemp()

    try:
        # Clone
        run_git(["git", "clone", repo_url, "."], cwd=temp_dir)
        run_git(["git", "fetch", "--all"], cwd=temp_dir)

        # Create or checkout branch
        remote_branches = run_git(["git", "branch", "-r"], cwd=temp_dir)
        if f"origin/{branch_name}" in remote_branches:
            run_git(["git", "checkout", branch_name], cwd=temp_dir)
            run_git(["git", "pull", "origin", branch_name], cwd=temp_dir)
        else:
            run_git(["git", "checkout", "-b", branch_name], cwd=temp_dir)

        # Configure git identity
        run_git(["git", "config", "user.email", "ai-sdlc@bot.com"], cwd=temp_dir)
        run_git(["git", "config", "user.name", "AI SDLC Bot"], cwd=temp_dir)

        # Write files
        files_written = 0
        for file in files:
            file_path = file["file_path"].replace("\\", os.sep)
            full_path = os.path.join(temp_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(file["content"])
            files_written += 1

        # Commit and push
        status = run_git(["git", "status", "--porcelain"], cwd=temp_dir)
        if not status:
            return {"status": "NO_CHANGES", "branch": branch_name}

        run_git(["git", "add", "."], cwd=temp_dir)
        run_git(["git", "commit", "-m", commit_message], cwd=temp_dir)
        run_git(["git", "push", "origin", branch_name, "--force"], cwd=temp_dir)

        return {
            "status": "PUSH_SUCCESS",
            "branch": branch_name,
            "files_pushed": files_written
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_github_pr(
    repo_name: str,
    branch_name: str,
    title: str,
    body: str
) -> dict:
    """Create a GitHub PR via API."""
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{repo_name}/pulls"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    payload = {
        "title": title,
        "head": branch_name,
        "base": "main",
        "body": body
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 201:
        return response.json()

    # PR already exists
    if response.status_code == 422 and "already exists" in response.text:
        pr_list_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{repo_name}/pulls?head={GITHUB_OWNER}:{branch_name}&state=open"
        pr_resp = requests.get(pr_list_url, headers=headers)
        prs = pr_resp.json()
        if prs:
            return prs[0]

    raise Exception(f"GitHub PR creation failed: {response.text}")


# -----------------------------------------
# Nodes
# -----------------------------------------

def push_code(state: DeliveryState) -> DeliveryState:
    """Push all generated changes to GitHub branch."""
    print("\n[Phase 6] Pushing code to GitHub...")

    branch_name = state["branch_name"]
    repo_url = state["repo_url"]

    # Prepare all files to push
    all_files = []

    # Generated code changes
    for change in state["generated_changes"]:
        all_files.append({
            "file_path": change["file_path"],
            "content": change["content"]
        })

    # Generated test files
    for test_file in state.get("test_files", []):
        if test_file.get("content"):
            all_files.append({
                "file_path": test_file["test_file_path"],
                "content": test_file["content"]
            })

    print(f"  Pushing {len(all_files)} files to branch: {branch_name}")

    try:
        push_result = push_files_to_branch(
            repo_url=repo_url,
            branch_name=branch_name,
            files=all_files,
            commit_message=f"feat: AI-generated changes — {state['requirement'][:60]}"
        )
        print(f"  ✅ Push result: {push_result['status']}")
        return {**state, "status": "CODE_PUSHED"}

    except Exception as e:
        print(f"  ❌ Push failed: {e}")
        return {**state, "status": "PUSH_FAILED"}


def create_pr(state: DeliveryState) -> DeliveryState:
    """Create GitHub PR for the changes."""
    print("\n[Phase 6] Creating GitHub PR...")

    if state["status"] == "PUSH_FAILED":
        return {**state, "status": "PR_SKIPPED"}

    branch_name = state["branch_name"]
    repo_url = state["repo_url"]

    # Extract repo name from URL
    repo_name = repo_url.rstrip(".git").split("/")[-1]

    # Build PR body
    changes_summary = "\n".join([
        f"- `{c['file_path']}`: {c.get('change_summary', '')[:80]}"
        for c in state["generated_changes"]
    ])

    pr_body = f"""## AI-Generated Changes

**Requirement:** {state['requirement']}

## Files Changed
{changes_summary}

## Tests Added
{len(state.get('test_files', []))} test file(s) generated

## Generated by
SDLC Automation Platform V2 — LangGraph Pipeline

---
*This PR was automatically generated. Please review carefully before merging.*
"""

    try:
        pr = create_github_pr(
            repo_name=repo_name,
            branch_name=branch_name,
            title=f"feat: {state['requirement'][:60]}",
            body=pr_body
        )

        pr_url = pr.get("html_url", "")
        print(f"  ✅ PR created: {pr_url}")

        return {
            **state,
            "pr_urls": [pr_url],
            "status": "PR_CREATED"
        }

    except Exception as e:
        print(f"  ❌ PR creation failed: {e}")
        return {**state, "status": "PR_FAILED"}


def human_approval_gate(state: DeliveryState) -> DeliveryState:
    """Human reviews the PR before merge is approved."""
    print("\n[Phase 6] ⏸ Waiting for PR review approval...")

    for url in state.get("pr_urls", []):
        print(f"  PR to review: {url}")

    human_input = interrupt("Review the PR and approve or reject")

    approved = False
    feedback = ""
    if isinstance(human_input, dict):
        approved = human_input.get("approved", False)
        feedback = human_input.get("feedback", "")

    return {
        **state,
        "approved": approved,
        "human_feedback": feedback,
        "status": "WAITING_FOR_PR_APPROVAL"
    }


def process_approval(state: DeliveryState) -> DeliveryState:
    if state.get("approved"):
        print(f"\n[Phase 6] ✅ PR approved — ready for deployment")
        return {**state, "status": "APPROVED_FOR_DEPLOYMENT"}
    else:
        print(f"\n[Phase 6] ❌ PR rejected — {state.get('human_feedback', '')}")
        return {**state, "status": "PR_REJECTED"}


def route_after_approval(state: DeliveryState) -> str:
    if state["status"] == "APPROVED_FOR_DEPLOYMENT":
        return "approved"
    return "rejected"


# -----------------------------------------
# Build Graph
# -----------------------------------------

def build_delivery_graph():
    builder = StateGraph(DeliveryState)

    builder.add_node("push_code", push_code)
    builder.add_node("create_pr", create_pr)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("process_approval", process_approval)

    builder.set_entry_point("push_code")
    builder.add_edge("push_code", "create_pr")
    builder.add_edge("create_pr", "human_approval_gate")
    builder.add_edge("human_approval_gate", "process_approval")

    builder.add_conditional_edges(
        "process_approval",
        route_after_approval,
        {
            "approved": END,
            "rejected": END
        }
    )

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_approval_gate"]
    )


# -----------------------------------------
# Run
# -----------------------------------------

def run_delivery(
    requirement: str,
    generated_changes: list,
    test_files: list,
    repo_url: str,
    branch_name: str,
    thread_id: str = "thread-delivery"
) -> dict:
    graph = build_delivery_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = DeliveryState(
        requirement=requirement,
        generated_changes=generated_changes,
        test_files=test_files,
        branch_name=branch_name,
        repo_url=repo_url,
        pr_urls=[],
        human_feedback="",
        approved=False,
        status="STARTED"
    )

    print("\n" + "="*50)
    print("--- Starting Phase 6 — Delivery ---")
    print("="*50)

    result = graph.invoke(initial_state, config)

    print(f"\nStatus after interrupt: {result['status']}")
    for url in result.get("pr_urls", []):
        print(f"  PR: {url}")

    return graph, config, result


def resume_delivery(graph, config, approved: bool, feedback: str = "") -> dict:
    print(f"\n--- Resuming Phase 6 (approved={approved}) ---")
    result = graph.invoke(
        Command(resume={"approved": approved, "feedback": feedback}),
        config
    )
    print(f"Final status: {result['status']}")
    return result


if __name__ == "__main__":
    # Mock generated changes from Phase 4
    mock_changes = [
        {
            "file_path": "app/models.py",
            "content": """from pydantic import BaseModel
from typing import Optional

class LeaveRequest(BaseModel):
    employee_name: str
    leave_type: str
    start_date: str
    end_date: str
    reason: str
    balance: int = 20

class LeaveStatus(BaseModel):
    leave_id: str
    status: str
    leave_balance: Optional[int] = None

class Employee(BaseModel):
    employee_id: str
    name: str
    leave_balance: int = 20
""",
            "change_summary": "Added Employee model and balance fields",
            "new_symbols_added": ["Employee"],
            "existing_symbols_modified": ["LeaveRequest", "LeaveStatus"]
        }
    ]

    mock_test_files = [
        {
            "test_file_path": "tests/test_app_models.py",
            "content": """import pytest
from app.models import LeaveRequest, LeaveStatus, Employee

def test_leave_request_default_balance():
    req = LeaveRequest(
        employee_name="John",
        leave_type="sick",
        start_date="2026-01-01",
        end_date="2026-01-02",
        reason="sick"
    )
    assert req.balance == 20

def test_employee_default_balance():
    emp = Employee(employee_id="E001", name="John")
    assert emp.leave_balance == 20
"""
        }
    ]

    requirement = "Add leave balance tracker. Each employee gets 20 days per year."

    graph6, config6, result6 = run_delivery(
        requirement=requirement,
        generated_changes=mock_changes,
        test_files=mock_test_files,
        repo_url="https://github.com/AkashW45/leave-mgmt-backend.git",
        branch_name="feature/leave-balance-test",
        thread_id="test-delivery-1"
    )

    print(f"\nStatus after interrupt: {result6['status']}")
    for url in result6.get("pr_urls", []):
        print(f"  PR: {url}")

    print("\n--- Simulating Human PR Approval ---")
    final = resume_delivery(graph6, config6, approved=True, feedback="Looks good")

    print(f"\n✅ Phase 6 Test Complete")
    print(f"Status: {final['status']}")
    for url in final.get("pr_urls", []):
        print(f"PR: {url}")