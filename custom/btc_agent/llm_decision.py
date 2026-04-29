# custom/btc_agent/llm_decision.py

import json
import re
import time
from dataclasses import dataclass
from typing import Literal, Optional

import requests
from .config import get_llm_config
from .indicators import BtcFeatures
from .market_lookup import BtcUpDownMarket
from .network import (
    check_internet_connectivity,
    mask_proxy_url,
)

DecisionSide = Literal["UP", "DOWN", "NO_TRADE"]


class ConnectivityCheckFailed(RuntimeError):
    pass


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
        "Market reference:\n"
        f"- Price to beat USD: {market.settlement_threshold}\n"
        f"- Settlement rule: UP wins only if BTC finishes above {market.settlement_threshold}; "
        f"DOWN wins only if BTC finishes below {market.settlement_threshold}.\n\n"
        "BTC features:\n"
        f"- Current BTC price USD: {features.price_usd:.2f}\n"
        f"- Market window open price USD: {features.window_open_price:.2f}\n"
        f"- Percent change from market window open: {features.delta_pct_from_window_open * 100:.4f}%\n"
        f"- Trailing 5-minute open price USD: {features.trailing_5m_open_price:.2f}\n"
        f"- Percent change from trailing 5-minute open: {features.delta_pct_from_trailing_5m_open * 100:.4f}%\n"
        f"- Change from previous tick USD: {features.delta_from_previous_tick}\n"
        f"- RSI(14): {features.rsi_14}\n"
        f"- 1-minute momentum USD: {features.momentum_1m}\n"
        f"- Trailing 5-minute momentum USD: {features.momentum_5m}\n"
        f"- Trailing 5-minute volatility: {features.volatility_5m}\n\n"
        "Keep `reason` short and concrete.\n"
        "Return ONLY the JSON object described in the system message."
    )


def _build_compact_user_prompt(features: BtcFeatures, market: BtcUpDownMarket) -> str:
    return (
        f"BTC 5m market slug: {market.slug}\n"
        f"Price to beat USD: {market.settlement_threshold}\n"
        f"Current BTC price USD: {features.price_usd:.2f}\n"
        f"Window open price USD: {features.window_open_price:.2f}\n"
        f"Trailing 5-minute open USD: {features.trailing_5m_open_price:.2f}\n"
        f"Delta from window open pct: {features.delta_pct_from_window_open * 100:.4f}%\n"
        f"Delta from trailing 5-minute open pct: {features.delta_pct_from_trailing_5m_open * 100:.4f}%\n"
        f"Change from previous tick USD: {features.delta_from_previous_tick}\n"
        f"RSI(14): {features.rsi_14}\n"
        f"1-minute momentum USD: {features.momentum_1m}\n"
        f"Trailing 5-minute momentum USD: {features.momentum_5m}\n"
        f"Trailing 5-minute volatility: {features.volatility_5m}\n"
        "Settlement: UP wins only above the price to beat; DOWN wins only below it.\n"
        'Return one JSON object with keys: decision, confidence, max_price_to_pay, reason.'
    )


def _build_minimal_user_prompt(features: BtcFeatures, market: BtcUpDownMarket) -> str:
    return (
        f"BTC price: {features.price_usd:.2f}\n"
        f"Price to beat: {market.settlement_threshold}\n"
        f"Delta pct: {features.delta_pct_from_window_open * 100:.4f}%\n"
        f"RSI14: {features.rsi_14}\n"
        f"Momentum1m: {features.momentum_1m}\n"
        f"Momentum5m: {features.momentum_5m}\n"
        f"Volatility5m: {features.volatility_5m}\n"
        "UP wins above the price to beat. DOWN wins below it.\n"
        'Return one JSON object with keys: decision, confidence, max_price_to_pay, reason.'
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
    phase: str = "primary",
) -> None:
    outcome = "response" if success else "failed"
    phase_suffix = "" if phase == "primary" else f" [{phase}]"
    print(
        f"LLM attempt {attempt_number}/{total_attempts} "
        f"({engine}/{model}){phase_suffix} {outcome}: {_truncate_log_text(detail)}"
    )


def _print_llm_connection_config(
    engine: str,
    model: str,
    timeout_seconds: float,
    proxy_url: Optional[str],
) -> None:
    print("LLM connection:")
    print(f"  engine            = {engine}")
    print(f"  model             = {model}")
    print(f"  timeout_seconds   = {timeout_seconds:.1f}")
    print(f"  proxy             = {mask_proxy_url(proxy_url)}")


def _check_connectivity_after_llm_failure() -> None:
    is_connected, detail = check_internet_connectivity()
    print(f"Internet connectivity check: {detail}")
    if not is_connected:
        raise ConnectivityCheckFailed(detail)


def _direct_http_post(url: str, **kwargs) -> requests.Response:
    session = requests.Session()
    session.trust_env = False
    try:
        return session.post(url, **kwargs)
    finally:
        session.close()


def _request_openai_once(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float,
) -> str:
    proxy_url = None
    _print_llm_connection_config(
        "openai",
        model,
        timeout_seconds,
        proxy_url,
    )
    response = _direct_http_post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise RuntimeError("OpenAI returned no message content")
    return str(content)


def _request_openai_decision(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    fallback_user_prompt: Optional[str] = None,
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
            try:
                _check_connectivity_after_llm_failure()
            except ConnectivityCheckFailed as connectivity_exc:
                _print_llm_attempt_result(
                    "openai",
                    model,
                    attempt_number,
                    retry_attempts,
                    False,
                    str(exc),
                )
                raise RuntimeError(f"OpenAI request failed: {connectivity_exc}") from exc
            if fallback_user_prompt and fallback_user_prompt != user_prompt:
                _print_llm_attempt_result(
                    "openai",
                    model,
                    attempt_number,
                    retry_attempts,
                    False,
                    str(exc),
                    phase="primary",
                )
                try:
                    raw_text = _request_openai_once(
                        model=model,
                        api_key=api_key,
                        system_prompt=system_prompt,
                        user_prompt=fallback_user_prompt,
                        timeout_seconds=timeout_seconds,
                    )
                    _print_llm_attempt_result(
                        "openai",
                        model,
                        attempt_number,
                        retry_attempts,
                        True,
                        raw_text or "{}",
                        phase="fallback",
                    )
                    return raw_text
                except Exception as compact_exc:
                    last_error = compact_exc
                    _print_llm_attempt_result(
                        "openai",
                        model,
                        attempt_number,
                        retry_attempts,
                        False,
                        str(compact_exc),
                        phase="fallback",
                    )
            else:
                _print_llm_attempt_result(
                    "openai",
                    model,
                    attempt_number,
                    retry_attempts,
                    False,
                    str(exc),
                )
            if attempt_number >= retry_attempts:
                raise RuntimeError(f"OpenAI request failed: {last_error}") from last_error
            time.sleep(retry_timer_seconds)

    raise RuntimeError(f"OpenAI request failed: {last_error}")


def _request_gemini_decision(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float = 10.0,
    attempt_number: int = 1,
    total_attempts: int = 3,
) -> str:
    proxy_url = None
    _print_llm_connection_config(
        "gemini",
        model,
        timeout_seconds,
        proxy_url,
    )

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
    try:
        response = _direct_http_post(
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
            total_attempts,
            True,
            raw_text or detail,
        )
        return raw_text
    except requests.RequestException as exc:
        _print_llm_attempt_result(
            "gemini",
            model,
            attempt_number,
            total_attempts,
            False,
            str(exc),
        )
        _check_connectivity_after_llm_failure()
        raise RuntimeError(f"Gemini request failed: {exc}") from exc
    except ConnectivityCheckFailed:
        raise
    except RuntimeError as exc:
        _print_llm_attempt_result(
            "gemini",
            model,
            attempt_number,
            total_attempts,
            False,
            str(exc),
        )
        _check_connectivity_after_llm_failure()
        raise RuntimeError(f"Gemini request failed: {exc}") from exc


def _request_gemini_decision_with_parse_retry(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float = 10.0,
    retry_attempts: int = 3,
    retry_timer_seconds: float = 2.0,
) -> dict:
    last_error = None

    for attempt in range(retry_attempts):
        attempt_number = attempt + 1
        try:
            raw_text = _request_gemini_decision(
                model=model,
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_seconds=timeout_seconds,
                attempt_number=attempt_number,
                total_attempts=retry_attempts,
            )
            return _extract_json_payload(raw_text)
        except ValueError as exc:
            last_error = exc
            _print_llm_attempt_result(
                "gemini",
                model,
                attempt_number,
                retry_attempts,
                False,
                f"Incomplete or invalid JSON: {exc}",
                phase="invalid-json",
            )
            if attempt_number >= retry_attempts:
                raise RuntimeError(f"Gemini request failed: {exc}") from exc
            time.sleep(retry_timer_seconds)
        except RuntimeError as exc:
            last_error = exc
            if isinstance(exc, ConnectivityCheckFailed):
                raise RuntimeError(f"Gemini request failed: {exc}") from exc
            if attempt_number >= retry_attempts:
                raise
            time.sleep(retry_timer_seconds)

    raise RuntimeError(f"Gemini request failed: {last_error}")


def _coerce_config_value(raw_value: object, caster, default):
    try:
        return caster(raw_value)
    except (TypeError, ValueError):
        return default


def test_llm_connection() -> tuple[bool, str]:
    cfg = get_llm_config()
    system_prompt = (
        "You are a connection test for an automated trading agent. "
        'Respond with a single JSON object: {"status":"ok"}.'
    )
    user_prompt = 'Return exactly {"status":"ok"} and nothing else.'
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

    try:
        if cfg.engine == "openai":
            raw_text = _request_openai_decision(
                model=cfg.model,
                api_key=cfg.api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_seconds=api_connection_timeout_seconds,
                retry_attempts=api_connection_retry_attempts,
                retry_timer_seconds=max(api_connection_retry_timer_seconds, 0.0),
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
                retry_timer_seconds=max(api_connection_retry_timer_seconds, 0.0),
            )
        else:
            raise RuntimeError(f"Unsupported AI engine: {cfg.engine}")
    except Exception as exc:
        return False, str(exc)

    if str(data.get("status", "")).lower() != "ok":
        return False, f"Unexpected LLM connection test payload: {data}"

    return True, f"LLM connection test succeeded ({cfg.engine}/{cfg.model})"


def decide_trade(features: BtcFeatures, market: BtcUpDownMarket) -> LlmDecision:
    cfg = get_llm_config()
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(features, market)
    compact_user_prompt = _build_compact_user_prompt(features, market)
    minimal_user_prompt = _build_minimal_user_prompt(features, market)
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
                user_prompt=compact_user_prompt,
                fallback_user_prompt=minimal_user_prompt,
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
