"""Model registry, env-key loading, and run parameters.

API keys are read from the environment. A `.env` file (in bench/ or the cwd) is loaded into
the environment at import without overriding variables that are already set, so keys can be
kept in a gitignored file.

Each registry entry sets the highest reasoning level its API offers (Opus 4.8 effort=max,
GPT-5.5 xhigh, Gemini 3.1 Pro thinking_level=high, DeepSeek/GLM reasoning_effort=max, Kimi
thinking on). A model is disabled with "enabled": False, or a subset is selected on the CLI.
N_ITERATIONS (the generation cycle count) defaults to 3 and is overridable via
$BENCH_ITERATIONS or --iterations.
"""
from __future__ import annotations

import os
import pathlib


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from bench/.env or ./.env into os.environ (no override)."""
    for p in (pathlib.Path(__file__).parent / ".env", pathlib.Path.cwd() / ".env"):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# Generation cap. For reasoning models this budget is shared with hidden chain-of-thought,
# so it is set well above the ~1k-token prose target. Per-model overridable below.
_MAX_TOKENS = 16000

MODELS: list[dict] = [
    {
        "key": "opus-4.8",
        "provider": "anthropic",
        "model_id": "claude-opus-4-8",
        "api_key_env": "ANTHROPIC_API_KEY",
        "effort": "max",
        # 2x _MAX_TOKENS. For Anthropic models with extended thinking, max_tokens caps
        # thinking and text together, whereas for the other families it effectively caps the
        # prose: Gemini does not return its thoughts, and GLM and Kimi stream thinking into
        # the output but stay far under the ceiling (about 3.3k and 5.1k tokens per cell).
        # At 16k the self_refine rewrite (which carries the draft and the critique in
        # context) can spend the whole budget thinking and emit no text, with
        # stop_reason=max_tokens. 32k leaves a prose budget behind the thinking, so the text
        # budget is comparable across families even though the raw cap is not.
        "max_tokens": 2 * _MAX_TOKENS,
        "enabled": True,
    },
    {
        "key": "fable-5",
        "provider": "anthropic",
        "model_id": "claude-fable-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "effort": "max",
        "max_tokens": _MAX_TOKENS,
        "enabled": True,
    },
    {
        "key": "gpt-5.5",
        "provider": "openai_responses",
        "model_id": "gpt-5.5",  # pin a dated model id for a frozen run
        "api_key_env": "OPENAI_API_KEY",
        "reasoning_effort": "xhigh",  # max for gpt-5.5
        "max_tokens": _MAX_TOKENS,
        "enabled": True,
    },
    {
        "key": "gemini-3.1-pro",
        "provider": "google",
        "model_id": "gemini-3.1-pro-preview",  # the 3-pro preview id was retired
        "api_key_env": "GEMINI_API_KEY",
        "thinking_level": "high",  # max
        "max_tokens": _MAX_TOKENS,
        "enabled": True,
    },
    {
        "key": "kimi-k2.6",
        "provider": "openai_compat",
        "model_id": "kimi-k2.6",
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.ai/v1",  # .cn endpoint needs a .cn key
        "token_param": "max_completion_tokens",
        "extra_body": {"thinking": {"type": "enabled"}},  # on by default; no effort dial
        "include_usage": True,
        "max_tokens": _MAX_TOKENS,
        "enabled": True,
    },
    {
        "key": "deepseek-v4",
        "provider": "openai_compat",
        "model_id": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "token_param": "max_tokens",
        "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "max"},
        "include_usage": True,
        # 32k: at a 16k cap, 51 of the first 513 cells spent the whole budget on reasoning.
        "max_tokens": 2 * _MAX_TOKENS,
        "enabled": True,
    },
    {
        "key": "grok-4.3",
        "provider": "openai_compat",
        "model_id": "grok-4.3",  # reasoning built in, no effort dial
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "token_param": "max_tokens",
        "include_usage": True,
        "max_tokens": _MAX_TOKENS,
        "enabled": True,
    },
    {
        "key": "grok-4.5",
        "provider": "openai_compat",
        "model_id": "grok-4.5",  # the dot matters: grok-4-5 returns 404
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "token_param": "max_tokens",
        "include_usage": True,
        "max_tokens": _MAX_TOKENS,
        "enabled": True,
    },
    {
        "key": "fugu-ultra",
        "provider": "openai_compat",
        "model_id": "fugu-ultra",  # Sakana Fugu Ultra
        "api_key_env": "SAKANA_API_KEY",
        "base_url": "https://api.sakana.ai/v1",
        "token_param": "max_tokens",
        "extra_body": {"reasoning": {"effort": "xhigh"}},  # high | xhigh ("max" aliases xhigh)
        "include_usage": False,  # stream usage support unverified
        "max_tokens": _MAX_TOKENS,
        "enabled": True,
    },
    {
        "key": "glm-5.2",
        "provider": "openai_compat",
        "model_id": "glm-5.2",
        "api_key_env": "ZAI_API_KEY",
        "base_url": "https://api.z.ai/api/paas/v4/",  # .cn: open.bigmodel.cn + ZHIPUAI_API_KEY
        "token_param": "max_tokens",
        "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "max"},
        "include_usage": False,  # stream usage support unverified on z.ai
        "max_tokens": _MAX_TOKENS,
        "enabled": True,
    },
]

# Registered models that were not run, with the reason. Every entry in MODELS either has a
# benchmark leg or an entry here. The open-weights legs (Qwen3.6-27B, Gemma 3 27B) are not in
# MODELS; they run locally through bench/gen_vllm.py.
EXCLUDED_MODELS: dict[str, str] = {
    "fable-5": (
        "No published list price at the time of the runs, so the leg could not be costed "
        "(LIST_PRICE_USD in bench/run_batch.py refuses to price it); Claude Opus 4.8 is the "
        "Anthropic leg."
    ),
    "gpt-5.5": (
        "Not run, for cost: list price $5/$30 per Mtok and no OpenAI batch lane in the "
        "harness, so a leg would run at full rate (roughly $390). Gemini 3.1 Pro and Claude "
        "Opus 4.8 cover the closed frontier; the paper states the omission."
    ),
    "fugu-ultra": (
        "Registered (Sakana AI, api.sakana.ai) but never run; not part of the benchmark."
    ),
}

# Legs whose generation is in progress. A model is listed here only while a run is under way,
# and is then moved to a benchmark leg or to EXCLUDED_MODELS. It is a committed declaration,
# not inferred from bench/results/<key>.batchstate.json, which is gitignored and so absent in CI.
IN_FLIGHT_MODELS: dict[str, str] = {}

N_ITERATIONS = int(os.environ.get("BENCH_ITERATIONS", "3"))
