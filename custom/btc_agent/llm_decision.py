# custom/btc_agent/llm_decision.py

import json
import time
from dataclasses import dataclass
from typing import Literal

import requests
from openai import OpenAI  # v1 SDK

from .config import get_llm_config
from .indicators import BtcFeatures
from .market_lookup import BtcUpDownMarket

DecisionSide = Literal["UP", "DOWN", "NO_TRADE"]


@dataclass
class LlmDecision:
    side: DecisionSide
    confidence: float
    max_price_to_pay: float
    reason: str


_RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


def _build_system_prompt() -> str:
    return (
        "You are an automated trading decision assistant for a 5-minute Bitcoin "
        "up/down prediction market on Polymarket.\n"
        "You MUST respond with a single JSON object and nothing else.\n"
        "Schema:\n"
        "{\n"
        '  "decision": "UP" | "DOWN" | "NO_TRADE",\n'
        '  "confidence": number between 0 and 1,\n'
        '  "max_price_to_pay": number between 0 and 1,\n'
        '  "reason": string\n'
        "}\n"
        'Be conservative and prefer "NO_TRADE" when signals are weak.'
    )


def _build_user_prompt(features: BtcFeatures, market: BtcUpDownMarket) -> str:
    return (
        f"Market title: {market.title}\n"
        f"Market slug: {market.slug}\n\n"
        "BTC features:\n"
        f"- Current BTC price USD: {features.price_usd:.2f}\n"
        f"- Window open price USD: {features.window_open_price:.2f}\n"
        f"- Percent change from window open: {features.delta_pct_from_window_open * 100:.4f}%\n"
        f"- RSI(14): {features.rsi_14}\n"
        f"- 5-minute momentum: {features.momentum_5m}\n"
        f"- 5-minute volatility: {features.volatility_5m}\n\n"
        "Return ONLY the JSON object described in the system message."
    )


def _extract_json_payload(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if not cleaned:
        raise ValueError("Empty LLM response body")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fence_marker = "```"
    if fence_marker in cleaned:
        fenced_sections = cleaned.split(fence_marker)
        for section in fenced_sections:
            candidate = section.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError(f"Could not find JSON object in LLM response: {cleaned[:220]}")


def _looks_like_truncated_json(raw_text: str) -> bool:
    cleaned = raw_text.strip()
    if not cleaned:
        return False
    return cleaned.startswith("{") and cleaned.count("{") > cleaned.count("}")


def _request_openai_decision(model: str, api_key: str, system_prompt: str, user_prompt: str) -> str:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=256,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"


def _response_error_message(response: requests.Response) -> str:
    body = response.text.strip().replace("\n", " ")
    if len(body) > 300:
        body = body[:300]
    return f"HTTP {response.status_code}: {body or response.reason}"


def _request_gemini_decision(model: str, api_key: str, system_prompt: str, user_prompt: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "required": ["decision", "confidence", "max_price_to_pay", "reason"],
                "properties": {
                    "decision": {
                        "type": "STRING",
                        "enum": ["UP", "DOWN", "NO_TRADE"],
                    },
                    "confidence": {"type": "NUMBER"},
                    "max_price_to_pay": {"type": "NUMBER"},
                    "reason": {"type": "STRING"},
                },
            },
        },
    }
    last_error = None

    for attempt in range(3):
        try:
            response = requests.post(
                url,
                params={"key": api_key},
                json=payload,
                timeout=30,
            )
            if response.status_code in _RETRYABLE_HTTP_STATUS_CODES and attempt < 2:
                last_error = RuntimeError(
                    f"Gemini request failed: {_response_error_message(response)}"
                )
                time.sleep(1.5 * (attempt + 1))
                continue

            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= 2:
                raise RuntimeError(f"Gemini request failed: {exc}") from exc
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"Gemini request failed: {last_error}")

    response_payload = response.json()
    candidates = response_payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    text_parts = [str(part.get("text", "")) for part in parts if part.get("text")]
    if not text_parts:
        raise RuntimeError("Gemini returned no text content")
    return "\n".join(text_parts)


def _request_gemini_decision_with_parse_retry(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
) -> dict:
    raw_text = _request_gemini_decision(
        model=model,
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    try:
        return _extract_json_payload(raw_text)
    except ValueError:
        if _looks_like_truncated_json(raw_text):
            retry_prompt = (
                "Return exactly one complete minified JSON object with keys "
                '"decision","confidence","max_price_to_pay","reason". '
                "No markdown. No preamble. No explanation."
            )
        else:
            retry_prompt = (
                f"{user_prompt}\n\n"
                "IMPORTANT: Return exactly one minified JSON object. "
                "Do not include markdown fences. "
                "Do not include introductory text."
            )
        retry_text = _request_gemini_decision(
            model=model,
            api_key=api_key,
            system_prompt=system_prompt,
            user_prompt=retry_prompt,
        )
        return _extract_json_payload(retry_text)


def decide_trade(features: BtcFeatures, market: BtcUpDownMarket) -> LlmDecision:
    cfg = get_llm_config()
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(features, market)

    try:
        if cfg.engine == "openai":
            raw_text = _request_openai_decision(
                model=cfg.model,
                api_key=cfg.api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            data = _extract_json_payload(raw_text)
        elif cfg.engine == "gemini":
            data = _request_gemini_decision_with_parse_retry(
                model=cfg.model,
                api_key=cfg.api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        else:
            raise RuntimeError(f"Unsupported AI engine: {cfg.engine}")
    except Exception as exc:
        return LlmDecision(
            side="NO_TRADE",
            confidence=0.0,
            max_price_to_pay=0.0,
            reason=f"LLM request failed ({cfg.engine}/{cfg.model}): {str(exc)[:220]}",
        )

    side = str(data.get("decision", "NO_TRADE")).upper()
    if side not in ("UP", "DOWN", "NO_TRADE"):
        side = "NO_TRADE"

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    try:
        max_price_to_pay = float(data.get("max_price_to_pay", 0.0))
    except Exception:
        max_price_to_pay = 0.0

    reason = str(data.get("reason", ""))[:300]

    return LlmDecision(
        side=side,
        confidence=confidence,
        max_price_to_pay=max_price_to_pay,
        reason=reason,
    )
