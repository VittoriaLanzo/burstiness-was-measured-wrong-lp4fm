"""Provider adapters behind one call: generate(model, system, user) -> (text, reasoning, usage).

Four backends cover the registry:
  - anthropic         : Claude            (streamed; adaptive thinking + the registry effort)
  - openai_responses  : GPT-5.5           (Responses API; reasoning effort=xhigh; streamed)
  - openai_compat     : Kimi / DeepSeek / Grok / GLM / Fugu (OpenAI chat API at a custom
                        base_url; streamed)
  - google            : Gemini 3.1 Pro    (google-genai; thinking_level=high)

SDKs are imported lazily, so only the libraries for the models being run are needed. A
missing SDK or key is reported by preflight() (no API call) and raised as a RuntimeError with
a pip hint at call time. Errors are not caught here; they propagate to the driver, which logs
them and fails the leg. An empty system prompt (the control mode) is omitted, not sent blank.
"""
from __future__ import annotations

import importlib
import os

# provider -> (import path, pip package)
SDK = {
    "anthropic": ("anthropic", "anthropic"),
    "openai_responses": ("openai", "openai"),
    "openai_compat": ("openai", "openai"),
    "google": ("google.genai", "google-genai"),
}


def _imp(module: str, pip: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise RuntimeError(f"{pip} not installed -- run `pip install {pip}` ({e})") from e


def preflight(model: dict) -> list[str]:
    """Problems that would stop this leg, found without any API call: missing SDK / missing key."""
    problems: list[str] = []
    mod, pip = SDK.get(model["provider"], (None, None))
    if mod is None:
        problems.append(f"unknown provider '{model['provider']}'")
    else:
        try:
            importlib.import_module(mod)
        except Exception as e:
            problems.append(f"SDK '{mod}' not importable -> pip install {pip} "
                            f"({type(e).__name__}: {e})")
    if not os.environ.get(model["api_key_env"]):
        problems.append(f"API key not set -> export {model['api_key_env']}")
    return problems


def generate(model: dict, system: str, user: str) -> tuple[str, str | None, dict | None]:
    p = model["provider"]
    if p == "anthropic":
        return _anthropic(model, system, user)
    if p == "openai_responses":
        return _openai_responses(model, system, user)
    if p == "openai_compat":
        return _openai_compat(model, system, user)
    if p == "google":
        return _google(model, system, user)
    raise ValueError(f"unknown provider: {p}")


def _anthropic(m, system, user):
    anthropic = _imp("anthropic", "anthropic")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    kw = {"system": system} if system else {}
    with client.messages.stream(
        model=m["model_id"],
        max_tokens=m["max_tokens"],
        thinking={"type": "adaptive"},
        output_config={"effort": m["effort"]},
        messages=[{"role": "user", "content": user}],
        **kw,
    ) as stream:
        msg = stream.get_final_message()
    text = "".join(b.text for b in msg.content if b.type == "text")
    reasoning = "".join(getattr(b, "thinking", "") for b in msg.content if b.type == "thinking") or None
    usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
    return text, reasoning, usage


def _openai_responses(m, system, user):
    openai = _imp("openai", "openai")
    client = openai.OpenAI(api_key=os.environ[m["api_key_env"]])
    kw = {"instructions": system} if system else {}
    with client.responses.stream(
        model=m["model_id"],
        reasoning={"effort": m["reasoning_effort"]},
        max_output_tokens=m["max_tokens"],
        input=user,
        **kw,  # no temperature/top_p: unsupported on reasoning models
    ) as stream:
        final = stream.get_final_response()
    usage = None
    if getattr(final, "usage", None):
        usage = {"input_tokens": getattr(final.usage, "input_tokens", None),
                 "output_tokens": getattr(final.usage, "output_tokens", None)}
    return final.output_text, None, usage


def _openai_compat(m, system, user):
    openai = _imp("openai", "openai")
    client = openai.OpenAI(api_key=os.environ[m["api_key_env"]], base_url=m["base_url"], timeout=600)
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
    kwargs = {m["token_param"]: m["max_tokens"]}
    if m.get("include_usage"):
        kwargs["stream_options"] = {"include_usage": True}
    stream = client.chat.completions.create(
        model=m["model_id"], messages=messages, stream=True,
        extra_body=m.get("extra_body") or None, **kwargs,
        # no temperature/top_p: Kimi rejects non-default temp; DeepSeek/GLM ignore it
    )
    text, reasoning, usage = [], [], None
    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = {"input_tokens": chunk.usage.prompt_tokens,
                     "output_tokens": chunk.usage.completion_tokens}
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        if getattr(d, "reasoning_content", None):
            reasoning.append(d.reasoning_content)
        if d.content:
            text.append(d.content)
    return "".join(text), ("".join(reasoning) or None), usage


def _google(m, system, user):
    genai = _imp("google.genai", "google-genai")
    types = importlib.import_module("google.genai.types")
    client = genai.Client(api_key=os.environ.get(m["api_key_env"]))
    cfg = dict(
        temperature=1.0,  # Gemini 3 degrades below default; keep at 1.0
        max_output_tokens=m["max_tokens"],
        thinking_config=types.ThinkingConfig(thinking_level=m["thinking_level"]),
        safety_settings=[
            types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.OFF)
            for c in (
                types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            )
        ],
    )
    if system:
        cfg["system_instruction"] = system
    resp = client.models.generate_content(
        model=m["model_id"], contents=user,
        config=types.GenerateContentConfig(**cfg),
    )
    usage = None
    um = getattr(resp, "usage_metadata", None)
    if um:
        usage = {"input_tokens": getattr(um, "prompt_token_count", None),
                 "output_tokens": getattr(um, "candidates_token_count", None),
                 # thought tokens are billed but never returned as text; recorded so the
                 # leg's cost is complete
                 "thoughts_tokens": getattr(um, "thoughts_token_count", None)}
    return (resp.text or ""), None, usage
