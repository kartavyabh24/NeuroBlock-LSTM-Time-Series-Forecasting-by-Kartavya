import importlib.util
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


def get_provider_options():
    # Keep provider list safe when openai package is missing.
    has_openai = importlib.util.find_spec("openai") is not None
    return ["LM Studio", "IBM watsonx"] if has_openai else ["IBM watsonx"]


def _strip_codey_output(text: str) -> str:
    # Best-effort cleanup if model returns code/cells.
    t = (text or "").strip()

    # Remove fenced code blocks if any.
    t = re.sub(r"```.*?```", "", t, flags = re.DOTALL).strip()

    # If it looks like notebook/code, try to extract a human paragraph.
    bad_markers = ["# %%", "print(", "import ", "def ", "from ", "f\"\"\"", "plt.", "st."]
    if any(m in t for m in bad_markers):
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
        return t2 if len(t2) >= 40 else t

    return t


def _strip_leading_instruction_echo(text: str) -> str:
    # Remove common leading instruction lines that models sometimes echo.
    lines = (text or "").splitlines()
    if not lines:
        return (text or "").strip()

    patterns = [
        re.compile(r"^do not use (any )?code( or tables)?[.:\s]*$", re.IGNORECASE),
        re.compile(r"^do not use (any )?code or tables[.:\s]*$", re.IGNORECASE),
        re.compile(r"^return only plain english text[.:\s]*$", re.IGNORECASE),
        re.compile(r"^respond (only )?in plain english[.:\s]*$", re.IGNORECASE),
        re.compile(r"^use plain english[.:\s]*$", re.IGNORECASE),
    ]

    i = 0
    while i < len(lines):
        candidate = lines[i].strip()
        if not candidate:
            i += 1
            continue
        if any(p.match(candidate) for p in patterns):
            i += 1
            continue
        break

    cleaned = "\n".join(lines[i:]).strip()
    return cleaned if cleaned else (text or "").strip()


def _extract_watsonx_text(data: dict):
    # Extract text and debug fields from watsonx response.
    results = data.get("results", [])
    if not results:
        return "", "missing_results", None, None

    first = results[0] if isinstance(results[0], dict) else {}
    text = (first.get("generated_text") or "").strip()
    stop_reason = first.get("stop_reason") or "unknown"
    generated_tokens = first.get("generated_token_count")
    input_tokens = first.get("input_token_count")
    return text, stop_reason, generated_tokens, input_tokens


def _lmstudio_generate(model_name: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    try:
        from openai import OpenAI
    except Exception:
        raise RuntimeError("LM Studio requires openai package. Install: pip install openai")

    try:
        client = OpenAI(base_url = "http://localhost:1234/v1", api_key = "lm-studio")
        resp = client.chat.completions.create(
            model = model_name,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature = temperature,
        )
    except Exception as e:
        raise RuntimeError(
            f"LM Studio error: {e}. "
            f"Check LM Studio server at http://localhost:1234/v1 "
            f"and that model '{model_name}' exists."
        )

    if not resp or not getattr(resp, "choices", None):
        raise RuntimeError("LM Studio returned an empty response (no choices).")

    msg = resp.choices[0].message if resp.choices else None
    content = (msg.content or "").strip() if msg else ""
    if not content:
        raise RuntimeError("LM Studio returned an empty message content.")

    cleaned = _strip_codey_output(content)
    cleaned = _strip_leading_instruction_echo(cleaned)
    if not cleaned.strip():
        raise RuntimeError("LM Studio returned empty generated text after cleanup.")

    return cleaned


def _watsonx_env():
    # Read IBM watsonx credentials from environment.
    api_key = os.getenv("IBM_API_KEY") or os.getenv("WATSONX_APIKEY") or ""
    project_id = os.getenv("IBM_PROJECT_ID") or os.getenv("WATSONX_PROJECT_ID") or ""
    space_id = os.getenv("IBM_SPACE_ID") or os.getenv("WATSONX_SPACE_ID") or ""
    base_url = (os.getenv("IBM_URL") or os.getenv("WATSONX_URL") or "https://us-south.ml.cloud.ibm.com").rstrip("/")

    if not api_key or (not project_id and not space_id):
        raise RuntimeError(
            "IBM watsonx credentials are not set. "
            "Set IBM_API_KEY + IBM_PROJECT_ID (or IBM_SPACE_ID). "
            "Equivalent WATSONX_* env vars are also supported."
        )

    return api_key, project_id, space_id, base_url


def _watsonx_generate(model_name: str, attempts: list) -> str:
    try:
        import requests
    except Exception:
        raise RuntimeError("IBM watsonx requires requests package. Install: pip install requests")

    api_key, project_id, space_id, base_url = _watsonx_env()
    request_timeout = 30

    try:
        iam_response = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers = {"Content-Type": "application/x-www-form-urlencoded"},
            data = {
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": api_key,
            },
            timeout = request_timeout,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"IAM request failed (network/timeout): {e}")

    if iam_response.status_code != 200:
        raise RuntimeError(f"IAM authentication failed: {iam_response.text}")

    access_token = iam_response.json().get("access_token")
    if not access_token:
        raise RuntimeError("IAM authentication failed: access_token missing in response.")

    url = f"{base_url}/ml/v1/text/generation?version=2023-05-29"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    last_error = "IBM watsonx failed with an unknown error."
    for idx, attempt in enumerate(attempts, start = 1):
        payload = {
            "model_id": model_name,
            "input": attempt["input"],
            "parameters": attempt["parameters"],
        }
        if project_id:
            payload["project_id"] = project_id
        else:
            payload["space_id"] = space_id

        try:
            response = requests.post(url, headers = headers, json = payload, timeout = request_timeout)
        except requests.RequestException as e:
            last_error = f"IBM watsonx request failed on attempt {idx} (network/timeout): {e}"
            continue

        if response.status_code == 200:
            data = response.json()
            text, stop_reason, generated_tokens, input_tokens = _extract_watsonx_text(data)

            if text:
                cleaned = _strip_codey_output(text)
                cleaned = _strip_leading_instruction_echo(cleaned)
                if cleaned.strip():
                    return cleaned

            last_error = (
                "IBM watsonx returned empty generated_text "
                f"(attempt={idx}, stop_reason={stop_reason}, "
                f"input_tokens={input_tokens}, generated_tokens={generated_tokens})."
            )
            continue

        if response.status_code in (400, 401, 403, 404):
            raise RuntimeError(f"IBM watsonx request failed with status {response.status_code}: {response.text}")

        last_error = f"IBM watsonx error on attempt {idx} (status={response.status_code}): {response.text}"

    raise RuntimeError(last_error)


def _get_requested_word_limit(user_question: str):
    q = str(user_question or "")
    match = re.search(r"\bin\s+(\d{1,3})\s+words?\b", q, flags = re.IGNORECASE)
    if not match:
        match = re.search(r"\b(\d{1,3})\s+words?\b", q, flags = re.IGNORECASE)
    if not match:
        return None
    limit = int(match.group(1))
    return max(1, min(120, limit))


def _apply_word_limit(answer_text: str, word_limit):
    text = str(answer_text or "").strip()
    if not word_limit:
        return text

    words = re.findall(r"\S+", text)
    if len(words) == word_limit:
        return " ".join(words).strip()

    if len(words) > word_limit:
        return " ".join(words[:word_limit]).strip()

    filler_words = re.findall(r"\S+", "Based on the displayed chart context and model assumptions only.")
    needed = word_limit - len(words)
    repeat_count = (needed // len(filler_words)) + 1
    padded_words = words + (filler_words * repeat_count)
    return " ".join(padded_words[:word_limit]).strip()


def explain(provider: str, prompt: str, model_name: str) -> str:
    # provider: "LM Studio" or "IBM watsonx"
    # prompt: prompt string
    # model_name: model name depending on provider
    safe_prompt = (
        "Here is the data for you to explain.\n"
        "Explain what the forecast implies, what the percent change means, and 1-2 risks/limitations.\n"
        "DATA START\n"
        f"{prompt.strip()}\n"
        "DATA END"
    )

    if provider == "LM Studio":
        return _lmstudio_generate(
            model_name = model_name,
            system_prompt = SYSTEM_RULES,
            user_prompt = safe_prompt,
            temperature = 0.2,
        )

    if provider == "IBM watsonx":
        attempts = [
            {
                "input": f"{SYSTEM_RULES}\n\n{safe_prompt}",
                "parameters": {
                    "temperature": 0.1,
                    "max_new_tokens": 280,
                },
            },
            {
                "input": (
                    f"{safe_prompt}\n\n"
                    "Respond with 4-6 plain English sentences."
                ),
                "parameters": {
                    "temperature": 0.05,
                    "max_new_tokens": 220,
                },
            },
            {
                "input": (
                    f"{safe_prompt}\n\n"
                    "Give at least 3 short sentences. Do not leave the answer empty."
                ),
                "parameters": {
                    "temperature": 0.0,
                    "max_new_tokens": 220,
                },
            },
        ]
        return _watsonx_generate(model_name = model_name, attempts = attempts)

    raise RuntimeError(f"Unknown provider: {provider}")


def safe_explain(provider: str, prompt: str, model_name: str) -> str:
    # Retry once with a simpler output instruction if first attempt fails.
    prompt_candidates = [
        prompt,
        prompt + "\n\nReply in plain English in 4-6 short sentences. Do not return an empty response.",
    ]
    last_error = "LLM did not return a valid response."

    for prompt_item in prompt_candidates:
        try:
            answer = explain(provider, prompt_item, model_name)
            if answer and str(answer).strip():
                return str(answer).strip()
            last_error = "LLM returned an empty response."
        except Exception as e:
            last_error = str(e)

    raise RuntimeError(last_error)


def open_chat_answer(provider_name: str, user_question: str, model_id: str, context_text: str) -> str:
    question = str(user_question or "").strip()
    if not question:
        raise RuntimeError("Question is empty.")

    word_limit = _get_requested_word_limit(question)
    word_rule = (
        f"User requested {word_limit} words. Use exactly {word_limit} words."
        if word_limit else
        "Answer directly and concisely."
    )

    system_prompt = (
        "You are a helpful assistant.\n"
        "Answer only the user's question directly.\n"
        "Do not rewrite the whole forecast unless user asks for it.\n"
        f"{word_rule}\n"
        "Use plain English."
    )

    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Optional forecast context:\n{context_text}\n\n"
        "Give only the final answer."
    )

    if provider_name == "LM Studio":
        answer = _lmstudio_generate(
            model_name = model_id,
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            temperature = 0.2,
        )
        return _apply_word_limit(answer, word_limit)

    if provider_name == "IBM watsonx":
        prompt_attempts = [
            {
                "input": f"{system_prompt}\n\n{user_prompt}",
                "parameters": {
                    "temperature": 0.1,
                    "max_new_tokens": 220,
                },
            },
            {
                "input": (
                    f"{system_prompt}\n\n"
                    "Do not add extra background. Answer only the question in final form.\n\n"
                    f"{user_prompt}"
                ),
                "parameters": {
                    "temperature": 0.05,
                    "max_new_tokens": 220,
                },
            },
        ]
        answer = _watsonx_generate(model_name = model_id, attempts = prompt_attempts)
        return _apply_word_limit(answer, word_limit)

    raise RuntimeError(f"Unknown provider: {provider_name}")