"""Model registry, env-key loading, and run parameters.

Keys are read from the environment only -- never hardcoded. As a convenience, a `.env`
file (in bench/ or the cwd) is loaded into the environment at import (existing vars win),
so you can keep keys in a gitignored file instead of exporting them each shell.

Reasoning is dialed to maximum on every model (Opus 4.8 effort=max, GPT-5.5 xhigh, Gemini
3.1 Pro thinking_level=high, DeepSeek/GLM reasoning_effort=max, Kimi thinking on). Disable
a model with "enabled": False, or select a subset on the CLI. N_ITERATIONS (the generation
cycle count) is config, never hardcoded, and is overridable via $BENCH_ITERATIONS or --iterations.
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
        # 2x _MAX_TOKENS, and this PRESERVES cross-family parity rather than breaking it.
        #
        # For Anthropic models with extended thinking, max_tokens caps thinking AND text
        # together. Everywhere else in this benchmark it effectively caps the prose: Gemini
        # never returns its thoughts, and GLM and Kimi stream thinking into output but landed
        # at ~3.3k and ~5.1k tokens per cell, far under the ceiling. So a flat 16k hands Opus
        # strictly LESS room to write than every other family gets.
        #
        # Measured 2026-07-13: at effort=max the self_refine rewrite (which carries the draft
        # and the critique in context) spent all 16,000 tokens thinking and emitted zero text.
        # Every self_refine cell in the smoke run came back empty with stop_reason=max_tokens.
        # 32k restores a real prose budget behind the thinking. It is a deviation the protocol
        # section must disclose, so the reviewer sees parity of TEXT budget, not of raw cap.
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
        "model_id": "gpt-5.5",  # pin "gpt-5.5-2026-04-23" for a frozen run
        "api_key_env": "OPENAI_API_KEY",
        "reasoning_effort": "xhigh",  # max for gpt-5.5
        "max_tokens": _MAX_TOKENS,
        "enabled": True,
    },
    {
        "key": "gemini-3.1-pro",
        "provider": "google",
        "model_id": "gemini-3.1-pro-preview",  # 3-pro-preview was retired; 3.1 is current
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
        # 32k, as run: at 16k, 51 of the first 513 cells spent the whole budget on reasoning.
        "max_tokens": 2 * _MAX_TOKENS,
        "enabled": True,
    },
    {
        "key": "grok-4.3",
        "provider": "openai_compat",
        "model_id": "grok-4.3",  # xAI's current flagship (1M ctx); reasoning built in, no effort dial
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
        "model_id": "grok-4.5",  # released 2026-07-08 ($2/$6 per M); the dot matters, grok-4-5 404s
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
        "model_id": "fugu-ultra",  # Sakana Fugu Ultra: multi-agent-as-a-model behind one API
        "api_key_env": "SAKANA_API_KEY",
        "base_url": "https://api.sakana.ai/v1",
        "token_param": "max_tokens",
        "extra_body": {"reasoning": {"effort": "xhigh"}},  # high | xhigh ("max" aliases xhigh)
        "include_usage": False,  # stream usage support unverified; flip on once confirmed
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
        "include_usage": False,  # stream usage support unverified on z.ai; flip on once confirmed
        "max_tokens": _MAX_TOKENS,
        "enabled": True,
    },
]

# Registered models we deliberately did not run, and why. A leg that is simply absent looks like
# an oversight to a reviewer; a leg that is absent with a stated reason is a limitation. Anything
# in MODELS without a benchmark leg belongs here or in the benchmark, never in neither.
#
# The Qwen exclusion (circularity: it was the synthetic-data generation family, so it would have
# sat on both sides of train/eval) was lifted 2026-07-02 when the T5 training was retired.
# Qwen3.6 is now benchmarked locally via bench/gen_vllm.py.
EXCLUDED_MODELS: dict[str, str] = {
    "fable-5": (
        "No published list price as of 2026-07-13, so the leg cannot be costed honestly "
        "(see LIST_PRICE_USD in bench/run_batch.py, which refuses to price it). Opus 4.8 "
        "carries the Anthropic flagship slot instead, which is the coverage the benchmark needs."
    ),
    "gpt-5.5": (
        "Cost, decided 2026-07-13. List price $5/$30 per Mtok (verified). The harness has no "
        "OpenAI batch lane, so it would run at full rate, roughly $390 against the ~$195 it "
        "would cost batched. With Gemini 3.1 Pro and Claude Opus 4.8 both in the benchmark, the "
        "closed-source frontier is represented. This is the one frontier absence and the paper "
        "discloses it in Limitations rather than leaving a reviewer to notice it."
    ),
    "fugu-ultra": (
        "Registered (Sakana AI, api.sakana.ai) but never run; it is not part of the "
        "benchmark, and no reason for dropping it was recorded."
    ),
}

# Legs being generated right now. A model sits here only while a run is genuinely in progress,
# and it must graduate to a benchmark leg or to EXCLUDED_MODELS. This is a COMMITTED declaration
# on purpose: the first version of the accountability check below inferred "in flight" from the
# presence of bench/results/<key>.batchstate.json, which is gitignored, so it passed locally and
# could never pass in CI. A guard that depends on untracked state is a guard that lies.
IN_FLIGHT_MODELS: dict[str, str] = {}

N_ITERATIONS = int(os.environ.get("BENCH_ITERATIONS", "3"))
