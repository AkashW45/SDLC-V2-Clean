from openai import OpenAI
import os, json
from agents.prompts.system_prompts import CRITIC_SYSTEM

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

def critique(artifact: dict, artifact_type: str, scope_contract: dict,
             original_requirement: str, prior_artifacts: dict = None) -> dict:
    user_msg = json.dumps({
        "artifact_type": artifact_type,
        "artifact": artifact,
        "scope_contract": scope_contract,
        "original_requirement": original_requirement,
        "prior_artifacts": prior_artifacts or {}
    }, indent=2)

    response = client.chat.completions.create(
        model="deepseek-v4-flash",  # use fast model for critic
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=4096
    )
    raw = response.choices[0].message.content
    if raw.strip().startswith("```"):
        raw = raw.strip().strip("```json").strip("```").strip()
    try:
        return json.loads(raw)
    except:
        return {"verdict": "ACCEPT", "scores": {}, "violations": []}