# llm_provider.py
from openai import OpenAI

def explain(provider: str, prompt: str, model_name: str) -> str:
    """
    provider: "LM Studio" (for now)
    prompt: your prompt string
    model_name: the model name shown in LM Studio local server
    """

    if provider != "LM Studio":
        raise ValueError("Only LM Studio is implemented right now. Choose 'LM Studio' in the app.")

    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio"  # any string works
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Educational only. Use only provided numbers. Be concise."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return resp.choices[0].message.content.strip()