# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Unified vision-LLM client for the sem_map pipeline.

A thin wrapper around the ``openai`` Python SDK so we can talk to any
OpenAI-compatible chat-completions endpoint with a single code path. This
mirrors the upstream SAM3 agent helper at
``sam3/agent/client_llm.py`` (https://github.com/facebookresearch/sam3).

Configure via env vars (preferred) or pass an explicit ``LLMConfig``:

    SEM_MAP_LLM_BASE_URL   e.g. https://generativelanguage.googleapis.com/v1beta/openai/
    SEM_MAP_LLM_MODEL      e.g. gemini-2.5-flash
    SEM_MAP_LLM_API_KEY    provider API key

Falls through to ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` so a vanilla
OpenAI install needs no extra config.

Provider examples:

    | Provider     | base_url                                                       | model                          |
    | ------------ | -------------------------------------------------------------- | ------------------------------ |
    | OpenAI       | https://api.openai.com/v1                                      | gpt-4o-mini                    |
    | Gemini       | https://generativelanguage.googleapis.com/v1beta/openai/       | gemini-2.5-flash               |
    | OpenRouter   | https://openrouter.ai/api/v1                                   | google/gemini-2.5-flash        |
    | vLLM local   | http://localhost:8001/v1                                       | Qwen/Qwen3-VL-8B-Thinking      |
    | Ollama local | http://localhost:11434/v1                                      | llava                          |
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Image helpers ────────────────────────────────────────────────────────────

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def get_image_base64_and_mime(image_path: str) -> tuple[Optional[str], Optional[str]]:
    """Read an image file and return ``(base64_str, mime_type)``.

    Mirrors the upstream SAM3 helper of the same name.
    """
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = _MIME_TYPES.get(ext, "image/jpeg")
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), mime_type
    except Exception as e:
        logger.warning("Failed to encode image %s: %s", image_path, e)
        return None, None


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Configuration for the unified vision-LLM client.

    ``max_tokens`` is the cap on output tokens. Default 8192 is sized to give
    even verbose thinking models (e.g. ``gemini-3.x-*-preview``,
    ``gpt-5-thinking``, ``qwen3-vl-*-thinking``) plenty of room to emit the
    visible JSON after internal reasoning. Non-thinking models like
    ``gemini-2.5-flash`` are unaffected — they only consume what they need.
    Override per-call by setting ``SEM_MAP_LLM_MAX_TOKENS``.
    """
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = 0.0
    # Provider-specific overrides forwarded as ``extra_body`` to the OpenAI
    # SDK. Useful for Gemini-native fields like ``generationConfig`` or
    # ``thinkingConfig`` that aren't part of the OpenAI schema.
    extra_body: Optional[dict] = None

    @classmethod
    def from_env(
        cls,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> "LLMConfig":
        """Build a config, preferring explicit args, then env vars.

        Recognized env vars (in priority order):
            base_url:    SEM_MAP_LLM_BASE_URL,  OPENAI_BASE_URL
            model:       SEM_MAP_LLM_MODEL
            api_key:     SEM_MAP_LLM_API_KEY,   OPENAI_API_KEY
            max_tokens:  SEM_MAP_LLM_MAX_TOKENS
        """
        env_max = os.environ.get("SEM_MAP_LLM_MAX_TOKENS")
        try:
            env_max_int = int(env_max) if env_max else None
        except ValueError:
            env_max_int = None
        resolved_max = max_tokens or env_max_int
        kwargs: dict = dict(
            base_url=(base_url
                      or os.environ.get("SEM_MAP_LLM_BASE_URL")
                      or os.environ.get("OPENAI_BASE_URL")),
            model=(model
                   or os.environ.get("SEM_MAP_LLM_MODEL")),
            api_key=(api_key
                     or os.environ.get("SEM_MAP_LLM_API_KEY")
                     or os.environ.get("OPENAI_API_KEY")),
        )
        if resolved_max is not None:
            kwargs["max_tokens"] = resolved_max
        return cls(**kwargs)

    def require(self) -> None:
        missing = [k for k, v in
                   {"model": self.model, "api_key": self.api_key}.items() if not v]
        if missing:
            raise ValueError(
                f"LLMConfig missing required field(s): {', '.join(missing)}. "
                "Set SEM_MAP_LLM_MODEL / SEM_MAP_LLM_API_KEY (and optionally "
                "SEM_MAP_LLM_BASE_URL) or pass them explicitly.")


# ── Core call ────────────────────────────────────────────────────────────────

def _process_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand ``{"type": "image", "image": <path>}`` blocks into base64 image_url.

    Same shape as upstream SAM3's ``send_generate_request``.
    """
    out = []
    for msg in messages:
        new_msg = dict(msg)
        content = msg.get("content")
        if msg.get("role") == "user" and isinstance(content, list):
            new_content = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "image":
                    img_path = str(c["image"]).replace("?", "%3F")
                    b64, mime = get_image_base64_and_mime(img_path)
                    if b64 is None:
                        logger.warning("Skipping unreadable image: %s", img_path)
                        continue
                    new_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": "high",
                        },
                    })
                else:
                    new_content.append(c)
            new_msg["content"] = new_content
        out.append(new_msg)
    return out


def call_vlm(
    messages: list[dict[str, Any]],
    config: LLMConfig,
    response_format: Optional[dict] = None,
) -> tuple[str, dict]:
    """Send a chat-completions request to any OpenAI-compatible endpoint.

    Args:
        messages: OpenAI-style messages. ``{"type": "image", "image": <path>}``
            content blocks are auto-encoded to base64 image_url.
        config: LLMConfig with base_url / model / api_key.
        response_format: Optional OpenAI ``response_format`` dict, e.g.
            ``{"type": "json_object"}``.

    Returns:
        ``(content, meta)`` where ``meta`` is a dict with at least
        ``finish_reason`` and ``usage`` (completion/prompt/total tokens),
        for diagnostics.

    Raises:
        Whatever the OpenAI SDK raises (NotFoundError, AuthenticationError,
        RateLimitError, APIConnectionError, ...). Callers are expected to
        catch and surface the exception; this helper deliberately does NOT
        swallow errors so root causes are visible.
    """
    config.require()
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "The 'openai' package is required for sem_map.perception.llm_client. "
            "Install it with: pip install 'openai>=1.40.0'") from e

    processed = _process_messages(messages)
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    # Use the legacy ``max_tokens`` field (not ``max_completion_tokens``).
    # ``max_completion_tokens`` is the newer OpenAI o-series field that
    # several compat endpoints (notably Gemini's OpenAI-compat) silently
    # ignore, falling back to a ~1024-token default that truncates responses.
    # Every OpenAI-compatible provider we care about (Gemini, vLLM, Ollama,
    # OpenRouter, Together, ...) understands ``max_tokens``.
    kwargs: dict[str, Any] = dict(
        model=config.model,
        messages=processed,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        n=1,
    )
    if response_format is not None:
        kwargs["response_format"] = response_format
    if config.extra_body:
        kwargs["extra_body"] = config.extra_body

    resp = client.chat.completions.create(**kwargs)
    if not resp.choices:
        raise RuntimeError(f"LLM returned no choices: {resp}")
    choice = resp.choices[0]
    content = choice.message.content
    finish_reason = getattr(choice, "finish_reason", None)
    usage = getattr(resp, "usage", None)
    meta = {
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        } if usage else None,
        "model": getattr(resp, "model", config.model),
    }
    if content is None:
        raise RuntimeError(f"LLM returned empty content (meta={meta})")
    return content, meta


# ── Class-presence check ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a strict visual classifier. Given an image and a list of candidate "
    "classes, decide for EACH class whether it is clearly visible in the image. "
    "Be conservative: only mark a class present if you can actually see it.\n\n"
    "For EVERY class you must provide a short (≤20 words) reason that justifies "
    "your decision based on what you see in the image:\n"
    "  - If present=true, describe the visual evidence that proves the class "
    "is there (e.g. 'large grey boulders in the foreground left of trail').\n"
    "  - If present=false, explain why you rejected it — what you looked for "
    "and did not find, or what made it ambiguous (e.g. 'no upright trunks "
    "visible, only low shrubs', 'surface is dirt not gravel').\n"
    "Never write generic placeholders like 'not visible' or 'visible'; the "
    "reason must be specific to this image.\n\n"
    "Return ONLY a single strict JSON object with the EXACT schema below. "
    "All property names and string values MUST be enclosed in double quotes. "
    "Do NOT wrap the response in markdown code fences. Do NOT include any "
    "prose, reasoning, or text before or after the JSON object.\n"
    '  {"classes": [{"id": <int>, "present": <bool>, "reason": "<short text>"}, ...]}\n'
    "You MUST return an entry for every candidate class id, in the same order."
)


def _extract_json_object(raw: str) -> str:
    """Pull a JSON object out of a possibly noisy LLM response.

    Handles common deviations:
      - markdown code fences (```json ... ``` or ``` ... ```)
      - prose / reasoning text before or after the JSON object
      - extra trailing junk after the closing brace

    Returns the substring between the first ``{`` and its matching ``}``.
    Raises ``ValueError`` if no balanced object can be found.
    """
    if not raw:
        raise ValueError("empty response")

    # Strip markdown code fences first.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        return fence.group(1)

    # Brace-balance scan from the first '{'.
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"no '{{' in response: {raw[:200]!r}")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    raise ValueError(
        f"unbalanced braces in response (len={len(raw)}, depth={depth} at end); "
        f"tail={raw[-200:]!r}")


def _build_user_prompt(non_bg: list[dict]) -> str:
    lines = ["Candidate classes:"]
    for c in non_bg:
        name = c.get("name", "")
        concept = c.get("concept", "") or name
        lines.append(f"  {c['id']}. {name} ({concept})")
    lines.append(
        "\nFor each class above, decide if it is clearly visible in the attached "
        "image and give a specific reason justifying acceptance or rejection. "
        'Return JSON of the form {"classes": [{"id": int, "present": bool, "reason": str}, ...]}.')
    return "\n".join(lines)


def check_class_presence(
    image_path: str,
    categories: list[dict],
    config: LLMConfig,
) -> list[dict]:
    """Ask the VLM which classes from ``categories`` appear in ``image_path``.

    Args:
        image_path: Path to an image file (jpg/png/etc.).
        categories: Seg-layer category dicts (id, name, concept, ...).
            Class id 0 is treated as background and skipped.
        config: LLMConfig.

    Returns:
        A list of decision dicts, one per non-background class:
            ``[{"id": int, "present": bool, "reason": str}, ...]``
        Each ``reason`` justifies why the class was accepted or rejected for
        this specific image. On any failure (HTTP error, JSON parse error,
        missing key) the result is fail-open: every class is marked
        ``present=True`` with a reason describing the fallback, so downstream
        SAM3 sees the same prompts it would have seen without the filter.
    """
    non_bg = [c for c in categories if c.get("id", 0) != 0]
    valid_ids = {c["id"] for c in non_bg}

    def _fail_open(reason: str) -> list[dict]:
        return [{"id": c["id"], "present": True, "reason": reason} for c in non_bg]

    if not non_bg:
        return []

    image_path = str(Path(image_path).resolve())
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": _build_user_prompt(non_bg)},
        ]},
    ]

    meta: dict = {}
    try:
        raw, meta = call_vlm(messages, config, response_format={"type": "json_object"})
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.warning("LLM request failed for %s: %s", image_path, err)
        return _fail_open(f"fail-open: {err}")

    try:
        json_str = _extract_json_object(raw)
        parsed = json.loads(json_str)
        items = parsed.get("classes", [])
        if not isinstance(items, list):
            raise ValueError(f"'classes' is not a list: {items!r}")
        by_id: dict[int, dict] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            cid = int(item["id"])
            if cid not in valid_ids:
                continue
            by_id[cid] = {
                "id": cid,
                "present": bool(item.get("present", False)),
                "reason": str(item.get("reason", "")).strip(),
            }
        # Ensure every non-bg class has an entry; fill missing ones conservatively.
        decisions = []
        for c in non_bg:
            cid = c["id"]
            if cid in by_id:
                decisions.append(by_id[cid])
            else:
                decisions.append({
                    "id": cid,
                    "present": True,
                    "reason": "fail-open: missing from LLM response",
                })
        return decisions
    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
        # Surface the raw model output (head + tail) so the user can debug what
        # the model actually returned and tune the prompt or switch models.
        # Also dump the FULL raw response next to the image for inspection.
        raw_text = raw or ""
        raw_len = len(raw_text)
        head = raw_text[:120].replace("\n", " ")
        tail = raw_text[-120:].replace("\n", " ")
        try:
            debug_path = Path(image_path).with_suffix(".llm_debug.txt")
            debug_path.write_text(raw_text)
            debug_note = f" full_raw_at={debug_path.name}"
        except OSError:
            debug_note = ""
        finish_reason = meta.get("finish_reason") if meta else None
        usage = meta.get("usage") if meta else None
        completion_tokens = (usage or {}).get("completion_tokens")
        meta_note = (f" finish_reason={finish_reason!r} "
                     f"completion_tokens={completion_tokens}")
        logger.warning(
            "LLM returned unparseable response for %s (%s, meta=%r); failing open. raw=%r",
            image_path, e, meta, raw_text)
        return _fail_open(
            f"fail-open: parse error ({e});{meta_note} raw_len={raw_len} "
            f"head={head!r} tail={tail!r}{debug_note}")
