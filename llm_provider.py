# llm_provider.py

from openai import OpenAI
import requests
import os
import re


SYSTEM_RULES = (
    "You are a finance-forecast explainer.\n"
    "Return ONLY plain English text (no code, no markdown, no bullet lists longer than 6 items).\n"
    "Do NOT output Python, Jupyter cells, 'print(', '# %%', triple quotes, or JSON.\n"
    "Use only the numeric values provided by the user.\n"
    "Keep it concise: 4-8 sentences max.\n"
    "End with: 'Educational use only (not financial advice).' "
)


def _strip_codey_output(text: str) -> str:
    """Best-effort cleanup if model returns code/cells."""
    t = (text or "").strip()

    # Remove fenced code blocks if any
    t = re.sub(r"```.*?```", "", t, flags=re.DOTALL).strip()

    # If it looks like notebook/code, try to extract a human paragraph
    bad_markers = ["# %%", "print(", "import ", "def ", "from ", "f\"\"\"", "plt.", "st."]
    if any(m in t for m in bad_markers):
        # Keep only lines that look like normal prose
        lines = [ln.strip() for ln in t.splitlines()]
        prose = []
        for ln in lines:
            if not ln:
                continue
            if any(ln.startswith(x) for x in ["#", "import", "from", "def", "print", "plt", "st", ")", "(", "%"]):
                continue
            if "```" in ln:
                continue
            prose.append(ln)

        t2 = " ".join(prose).strip()
        # Fallback: if we stripped too much, return original trimmed
        return t2 if len(t2) >= 40 else t

    return t


def explain(provider: str, prompt: str, model_name: str) -> str:
    """
    provider: "LM Studio" or "IBM watsonx"
    prompt: your prompt string
    model_name: model name depending on provider
    """

    # Wrap prompt so it does NOT resemble code
    safe_prompt = (
        "Here is the data for you to explain.\n"
        "Explain what the forecast implies, what the percent change means, and 1-2 risks/limitations.\n"
        "DATA START\n"
        f"{prompt.strip()}\n"
        "DATA END"
    )

    # -------------------------
    # LM Studio (Local)
    # -------------------------
    if provider == "LM Studio":
        try:
            client = OpenAI(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio"
            )

            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_RULES},
                    {"role": "user", "content": safe_prompt}
                ],
                temperature=0.2
            )

            if not resp or not getattr(resp, "choices", None):
                raise ValueError("LM Studio returned an empty response (no choices).")

            msg = resp.choices[0].message
            content = (msg.content or "").strip() if msg else ""
            if not content:
                raise ValueError("LM Studio returned an empty message content.")

            return _strip_codey_output(content)

        except Exception as e:
            raise RuntimeError(
                f"LM Studio error: {e}. "
                f"Check LM Studio server at http://localhost:1234/v1 "
                f"and that model '{model_name}' exists."
            )

    # -------------------------
    # IBM watsonx
    # -------------------------
    elif provider == "IBM watsonx":

        # Accept either IBM_* or WATSONX_* env vars
        IBM_API_KEY = os.getenv("IBM_API_KEY") or os.getenv("WATSONX_APIKEY") or ""
        IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID") or os.getenv("WATSONX_PROJECT_ID") or ""
        IBM_URL = os.getenv("IBM_URL") or os.getenv("WATSONX_URL") or "https://us-south.ml.cloud.ibm.com"

        if not IBM_API_KEY or not IBM_PROJECT_ID:
            raise ValueError(
                "IBM watsonx credentials are not set. "
                "Set IBM_API_KEY + IBM_PROJECT_ID (or WATSONX_APIKEY + WATSONX_PROJECT_ID)."
            )

        REQUEST_TIMEOUT = 30

        # Step 1: IAM token
        try:
            iam_response = requests.post(
                "https://iam.cloud.ibm.com/identity/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": IBM_API_KEY
                },
                timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException as e:
            raise RuntimeError(f"IAM request failed (network/timeout): {e}")

        if iam_response.status_code != 200:
            raise ValueError(f"IAM authentication failed: {iam_response.text}")

        access_token = iam_response.json().get("access_token")
        if not access_token:
            raise ValueError("IAM authentication failed: access_token missing in response.")

        # Step 2: watsonx text generation
        url = f"{IBM_URL}/ml/v1/text/generation?version=2023-05-29"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        # IMPORTANT: Many watsonx models behave better if you include "instruction" + "input"
        # but this endpoint expects "input". We'll embed system rules into the input.
        final_input = f"{SYSTEM_RULES}\n\n{safe_prompt}"

        payload = {
            "model_id": model_name,
            "input": final_input,
            "parameters": {
                "temperature": 0.2,
                "max_new_tokens": 250,
                "stop_sequences": ["```", "# %%", "print(", "import ", "def "]
            },
            "project_id": IBM_PROJECT_ID
        }

        last_error = None
        for _ in range(2):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as e:
                last_error = f"IBM watsonx request failed (network/timeout): {e}"
                continue

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if not results or "generated_text" not in results[0]:
                    raise ValueError(f"IBM watsonx returned unexpected response: {data}")

                text = (results[0].get("generated_text") or "").strip()
                if not text:
                    raise ValueError("IBM watsonx returned empty generated_text.")

                return _strip_codey_output(text)

            last_error = f"IBM watsonx error: {response.text}"

        raise ValueError(last_error or "IBM watsonx failed with an unknown error.")

    else:
        raise ValueError(f"Unknown provider: {provider}")