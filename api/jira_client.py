import os
import base64
import httpx


def fetch_jira_metadata(project_key: str) -> dict:
    """
    Fetch Jira project metadata and normalize into deterministic structure.
    Dynamically detects custom field IDs (e.g., Story Points).
    """

    email = os.getenv("JIRA_EMAIL")
    token = os.getenv("JIRA_API_TOKEN")
    domain = os.getenv("JIRA_BASE_URL")

    if not all([email, token, domain]):
        raise ValueError("JIRA environment variables not configured")

    credentials = base64.b64encode(
        f"{email}:{token}".encode()
    ).decode()

    headers = {
        "Authorization": f"Basic {credentials}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    # --- Fetch createmeta ---
    url = (
        f"https://{domain}/rest/api/3/issue/createmeta"
        f"?projectKeys={project_key}"
        f"&expand=projects.issuetypes.fields"
    )

    response = httpx.get(url, headers=headers, timeout=15)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch Jira metadata: {response.text}")

    data = response.json()

    if not data.get("projects"):
        raise Exception("Project not found in Jira metadata")

    project = data["projects"][0]

    issue_types = {}
    story_points_field_id = None

    # --- Parse issue types + dynamic fields ---
    for issue in project.get("issuetypes", []):
        issue_types[issue["name"]] = issue["id"]

        # Detect Story Points field dynamically for Story issue type
        if issue["name"] == "Story":
            fields = issue.get("fields", {})

            for field_id, field_data in fields.items():
                schema = field_data.get("schema", {})
                custom_type = schema.get("custom")

                if custom_type == "com.pyxis.greenhopper.jira:jsw-story-points":
                    story_points_field_id = field_id

    # --- Fetch priorities ---
    priorities_url = f"https://{domain}/rest/api/3/priority"
    priorities_response = httpx.get(priorities_url, headers=headers, timeout=15)

    if priorities_response.status_code != 200:
        raise Exception("Failed to fetch Jira priorities")

    priorities = {
        p["name"]: p["id"]
        for p in priorities_response.json()
    }

    return {
        "project_id": project["id"],
        "issue_types": issue_types,
        "priorities": priorities,
        "dynamic_fields": {
            "story_points": story_points_field_id
        }
    }