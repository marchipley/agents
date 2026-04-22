# custom/btc_agent/llm_decision.py

import json
import re
import time
from dataclasses import dataclass
from typing import Literal, Optional

import requests
from openai import OpenAI  # v1 SDK

from .config import get_llm_config
from .indicators import BtcFeatures
from .market_lookup import BtcUpDownMarket
from .network import get_proxy_url_for_httpx, http_post

DecisionSide = Literal["UP", "DOWN", "NO_TRADE"]


@dataclass
class LlmDecision:
    side: DecisionSide
    confidence: float
    max_price_to_pay: float
    reason: str


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
        "Keep the reason concise, ideally under 120 characters.\n"
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
        "Keep `reason` short and concrete.\n"
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

    partial_payload = _extract_partial_json_payload(cleaned)
    if partial_payload is not None:
        return partial_payload

    raise ValueError(f"Could not find JSON object in LLM response: {cleaned[:220]}")


def _extract_partial_json_payload(cleaned: str) -> Optional[dict]:
    if not cleaned.startswith("{"):
        return None

    decision_match = re.search(
        r'"decision"\s*:\s*"(UP|DOWN|NO_TRADE)"',
        cleaned,
        flags=re.IGNORECASE,
    )
    confidence_match = re.search(
        r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?)',
        cleaned,
    )
    max_price_match = re.search(
        r'"max_price_to_pay"\s*:\s*(-?\d+(?:\.\d+)?)',
        cleaned,
    )

    if not decision_match:
        return None

    decision = decision_match.group(1).upper()
    confidence = float(confidence_match.group(1)) if confidence_match else None
    max_price_to_pay = float(max_price_match.group(1)) if max_price_match else None

    if decision != "NO_TRADE" and (confidence is None or max_price_to_pay is None):
        return None

    if confidence is None:
        confidence = 0.0
    if max_price_to_pay is None:
        max_price_to_pay = 0.0

    reason = "Truncated LLM response"
    found_reason_field = False
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned, flags=re.DOTALL)
    if reason_match:
        found_reason_field = True
        reason = reason_match.group(1).strip() or reason
    else:
        truncated_reason_match = re.search(
            r'"reason"\s*:\s*"(.+)$',
            cleaned,
            flags=re.DOTALL,
        )
        if truncated_reason_match:
            found_reason_field = True
            reason = truncated_reason_match.group(1).strip() or reason

    if not found_reason_field and decision != "NO_TRADE":
        return None

    if not found_reason_field and confidence_match is not None and max_price_match is not None:
        return None

    if not found_reason_field:
        reason = "Truncated NO_TRADE response"

    return {
        "decision": decision,
        "confidence": confidence,
        "max_price_to_pay": max_price_to_pay,
        "reason": reason[:300],
    }


def _looks_like_truncated_json(raw_text: str) -> bool:
    cleaned = raw_text.strip()
    if not cleaned:
        return False
    return cleaned.startswith("{") and cleaned.count("{") > cleaned.count("}")


def _response_error_message(response: requests.Response) -> str:
    body = response.text.strip().replace("\n", " ")
    if len(body) > 300:
        body = body[:300]
    return f"HTTP {response.status_code}: {body or response.reason}"


def _truncate_log_text(text: str, limit: int = 240) -> str:
    condensed = text.strip().replace("\n", " ")
    if len(condensed) <= limit:
        return condensed
    return condensed[:limit]


def _print_llm_attempt_result(
    engine: str,
    model: str,
    attempt_number: int,
    total_attempts: int,
    success: bool,
    detail: str,
) -> None:
    outcome = "response" if success else "failed"
    print(
        f"LLM attempt {attempt_number}/{total_attempts} "
        f"({engine}/{model}) {outcome}: {_truncate_log_text(detail)}"
    )


def _request_openai_once(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float,
) -> str:
    client_kwargs = {
        "api_key": api_key,
        "timeout": timeout_seconds,
        "max_retries": 0,
    }
    proxy_url = get_proxy_url_for_httpx()
    if proxy_url:
        try:
            from openai import DefaultHttpxClient

            client_kwargs["http_client"] = DefaultHttpxClient(proxy=proxy_url)
        except Exception:
            pass

    client = OpenAI(**client_kwargs)
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


def _request_openai_decision(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float = 10.0,
    retry_attempts: int = 3,
    retry_timer_seconds: float = 2.0,
) -> str:
    last_error = None

    for attempt in range(retry_attempts):
        attempt_number = attempt + 1
        try:
            raw_text = _request_openai_once(
                model=model,
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_seconds=timeout_seconds,
            )
            _print_llm_attempt_result(
                "openai",
                model,
                attempt_number,
                retry_attempts,
                True,
                raw_text or "{}",
            )
            return raw_text
        except Exception as exc:
            last_error = exc
            _print_llm_attempt_result(
                "openai",
                model,
                attempt_number,
                retry_attempts,
                False,
                str(exc),
            )
            if attempt_number >= retry_attempts:
                raise RuntimeError(f"OpenAI request failed: {exc}") from exc
            time.sleep(retry_timer_seconds)

    raise RuntimeError(f"OpenAI request failed: {last_error}")


def _request_gemini_decision(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float = 10.0,
    retry_attempts: int = 3,
    retry_timer_seconds: float = 2.0,
) -> str:
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

    for attempt in range(retry_attempts):
        attempt_number = attempt + 1
        try:
            response = http_post(
                url,
                params={"key": api_key},
                json=payload,
                timeout=timeout_seconds,
            )
            detail = response.text.strip() or response.reason or "empty response"
            response.raise_for_status()
            response_payload = response.json()
            candidates = response_payload.get("candidates") or []
            if not candidates:
                raise RuntimeError("Gemini returned no candidates")

            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            text_parts = [str(part.get("text", "")) for part in parts if part.get("text")]
            if not text_parts:
                raise RuntimeError("Gemini returned no text content")
            raw_text = "\n".join(text_parts)
            _print_llm_attempt_result(
                "gemini",
                model,
                attempt_number,
                retry_attempts,
                True,
                raw_text or detail,
            )
            return raw_text
        except requests.RequestException as exc:
            last_error = exc
            _print_llm_attempt_result(
                "gemini",
                model,
                attempt_number,
                retry_attempts,
                False,
                str(exc),
            )
            if attempt_number >= retry_attempts:
                raise RuntimeError(f"Gemini request failed: {exc}") from exc
            time.sleep(retry_timer_seconds)
        except RuntimeError as exc:
            last_error = exc
            _print_llm_attempt_result(
                "gemini",
                model,
                attempt_number,
                retry_attempts,
                False,
                str(exc),
            )
            if attempt_number >= retry_attempts:
                raise RuntimeError(f"Gemini request failed: {exc}") from exc
            time.sleep(retry_timer_seconds)

    raise RuntimeError(f"Gemini request failed: {last_error}")


def _request_gemini_decision_with_parse_retry(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float = 10.0,
    retry_attempts: int = 3,
    retry_timer_seconds: float = 2.0,
) -> dict:
    raw_text = _request_gemini_decision(
        model=model,
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout_seconds=timeout_seconds,
        retry_attempts=retry_attempts,
        retry_timer_seconds=retry_timer_seconds,
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
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
            retry_timer_seconds=retry_timer_seconds,
        )
        return _extract_json_payload(retry_text)


def _coerce_config_value(raw_value: object, caster, default):
    try:
        return caster(raw_value)
    except (TypeError, ValueError):
        return default


def decide_trade(features: BtcFeatures, market: BtcUpDownMarket) -> LlmDecision:
    cfg = get_llm_config()
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(features, market)
    api_connection_timeout_seconds = _coerce_config_value(
        getattr(cfg, "api_connection_timeout_seconds", 10.0),
        float,
        10.0,
    )
    api_connection_retry_timer_seconds = _coerce_config_value(
        getattr(cfg, "api_connection_retry_timer_seconds", 2.0),
        float,
        2.0,
    )
    api_connection_retry_attempts = max(
        _coerce_config_value(getattr(cfg, "api_connection_retry_attempts", 3), int, 3),
        1,
    )
    api_connection_retry_timer_seconds = max(api_connection_retry_timer_seconds, 0.0)

    try:
        if cfg.engine == "openai":
            raw_text = _request_openai_decision(
                model=cfg.model,
                api_key=cfg.api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_seconds=api_connection_timeout_seconds,
                retry_attempts=api_connection_retry_attempts,
                retry_timer_seconds=api_connection_retry_timer_seconds,
            )
            data = _extract_json_payload(raw_text)
        elif cfg.engine == "gemini":
            data = _request_gemini_decision_with_parse_retry(
                model=cfg.model,
                api_key=cfg.api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_seconds=api_connection_timeout_seconds,
                retry_attempts=api_connection_retry_attempts,
                retry_timer_seconds=api_connection_retry_timer_seconds,
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
