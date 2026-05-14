import os
import time
from datetime import datetime
from openai import OpenAI

class LLMGateway:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        )
        self.telemetry_file = "token_telemetry.txt"

    def generate(self, prompt: str = None, messages: list = None, model: str = "deepseek-chat", temperature: float = 0.2, tag: str = "general", **kwargs) -> str:
        start_time = time.time()
        
        # Smart routing: Use 'messages' if provided, otherwise wrap 'prompt'
        if messages is None:
            if prompt is None:
                raise ValueError("Must provide either 'prompt' or 'messages'")
            messages = [{"role": "user", "content": prompt}]
            
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        
        end_time = time.time()
        latency = end_time - start_time

        prompt_tokens = getattr(response.usage, 'prompt_tokens', None)
        completion_tokens = getattr(response.usage, 'completion_tokens', None)
        total_tokens = getattr(response.usage, 'total_tokens', None)

        timestamp = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        log_line = (
            f"[{timestamp}] Phase/Step: {tag} | Model: {model} | Latency: {latency:.2f}s | "
            f"Tokens: {prompt_tokens} in / {completion_tokens} out ({total_tokens} total)\n"
        )

        try:
            with open(self.telemetry_file, "a") as f:
                f.write(log_line)
        except Exception:
            print(f"[LLM_GATEWAY] Failed to write telemetry to {self.telemetry_file}")

        print(f"[LLM_GATEWAY] Model: {model} | Latency: {latency:.2f}s")
        return response.choices[0].message.content

gateway = LLMGateway()