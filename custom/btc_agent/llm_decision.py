# custom/btc_agent/llm_decision.py

import json
import os
import re
import time
import threading
import uuid
from dataclasses import dataclass
from typing import Literal, Optional

import requests
import websocket
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


_OPENAI_REALTIME_CLIENT = None
_OPENAI_REALTIME_CLIENT_LOCK = threading.Lock()


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
        'Be conservative and prefer "NO_TRADE" when signals are weak.\n'
        "Your job is regime detection and directional confidence, not price-capping.\n"
        "Interpret confidence as the mathematical probability that your chosen side wins.\n"
        "Treat Window Delta as the primary confidence signal in the final 10 seconds.\n"
        "If Window Delta is below 0.005% near T-10, ignore TA noise and prefer NO_TRADE.\n"
        "If Window Delta is above 0.15% near T-10, confidence should usually be 0.95 or higher.\n"
        "If confidence is above 0.90, treat it as a directive to get in rather than demanding extra edge buffer.\n"
        "If time remaining is under 5 seconds and confidence is above 0.70, avoid NO_TRADE unless the signal is clearly invalid.\n"
        "`max_price_to_pay` is informational only and is not used by execution.\n"
        "For directional trades, set `max_price_to_pay` to 1.0 unless you have a strong reason not to.\n"
        "If Window Delta is above 0.15% near T-10, you may set `max_price_to_pay` as high as 0.97."
    )


def _build_openai_realtime_system_prompt() -> str:
    return (
        "Return one JSON object only with keys decision, confidence, max_price_to_pay, reason. "
        "decision must be UP, DOWN, or NO_TRADE. "
        "confidence is win probability 0..1. "
        "Use Window Delta as the strongest late signal. "
        "If abs(Window Delta) < 0.005 near T-10, prefer NO_TRADE. "
        "If abs(Window Delta) > 0.15 near T-10, confidence should usually be very high. "
        "max_price_to_pay is informational only; use 1.0 for directional trades."
    )


def _build_user_prompt(features: BtcFeatures, market: BtcUpDownMarket, up_snapshot=None, down_snapshot=None) -> str:
    time_remaining_seconds = max(market.end_ts - int(features.as_of.timestamp()), 0)
    gap_to_target = (
        None
        if market.settlement_threshold in (None, 0)
        else features.price_usd - market.settlement_threshold
    )
    required_velocity_to_win = (
        None
        if gap_to_target is None or time_remaining_seconds <= 0
        else abs(gap_to_target) / time_remaining_seconds
    )
    return (
        f"Market title: {market.title}\n"
        f"Market slug: {market.slug}\n\n"
        "Market reference:\n"
        f"- Price to beat USD: {market.settlement_threshold}\n"
        f"- Settlement rule: UP wins only if BTC finishes above {market.settlement_threshold}; "
        f"DOWN wins only if BTC finishes below {market.settlement_threshold}.\n"
        f"- Time remaining seconds: {time_remaining_seconds}\n"
        f"- Window Delta pct: {features.delta_pct_from_window_open * 100:.4f}%\n"
        f"- UP Polymarket ask/buy quote: {getattr(up_snapshot, 'buy_quote', None)}\n"
        f"- DOWN Polymarket ask/buy quote: {getattr(down_snapshot, 'buy_quote', None)}\n"
        f"- UP top-book imbalance: {getattr(up_snapshot, 'top_level_book_imbalance', None)}\n"
        f"- DOWN top-book imbalance: {getattr(down_snapshot, 'top_level_book_imbalance', None)}\n"
        f"- UP imbalance pressure: {getattr(up_snapshot, 'imbalance_pressure', None)}\n"
        f"- DOWN imbalance pressure: {getattr(down_snapshot, 'imbalance_pressure', None)}\n"
        f"- Required velocity to win USD/sec: {required_velocity_to_win}\n\n"
        "BTC features:\n"
        f"- Current BTC price USD: {features.price_usd:.2f}\n"
        f"- Market window open price USD: {features.window_open_price:.2f}\n"
        f"- Percent change from market window open: {features.delta_pct_from_window_open * 100:.4f}%\n"
        f"- Trailing 5-minute open price USD: {features.trailing_5m_open_price:.2f}\n"
        f"- Percent change from trailing 5-minute open: {features.delta_pct_from_trailing_5m_open * 100:.4f}%\n"
        f"- Change from previous tick USD: {features.delta_from_previous_tick}\n"
        f"- RSI(9): {features.rsi_9}\n"
        f"- RSI(14): {features.rsi_14}\n"
        f"- RSI speed divergence: {features.rsi_speed_divergence}\n"
        f"- 1-minute momentum USD: {features.momentum_1m}\n"
        f"- Trailing 5-minute momentum USD: {features.momentum_5m}\n"
        f"- Velocity over last 15 seconds USD: {features.velocity_15s}\n"
        f"- Velocity over last 30 seconds USD: {features.velocity_30s}\n"
        f"- Momentum acceleration: {features.momentum_acceleration}\n"
        f"- EMA(9): {features.ema_9}\n"
        f"- EMA(21): {features.ema_21}\n"
        f"- EMA alignment (Price > EMA9 > EMA21): {features.ema_alignment}\n"
        f"- EMA cross direction: {features.ema_cross_direction}\n"
        f"- ADX(14): {features.adx_14}\n"
        f"- ATR(14): {features.atr_14}\n"
        f"- Trailing 5-minute volatility: {features.volatility_5m}\n"
        f"- Consecutive flat ticks: {features.consecutive_flat_ticks}\n"
        f"- Consecutive directional ticks: {features.consecutive_directional_ticks}\n\n"
        "Decision policy:\n"
        "- Focus on regime detection and direction, not limit pricing.\n"
        "- Confidence should represent your estimated win probability for the chosen side.\n"
        "- Window Delta is the master confidence signal near T-10.\n"
        "- Treat order-book imbalance and imbalance pressure as leading indicators.\n"
        "- Do not fade PARABOLIC_UP or PARABOLIC_DOWN regimes just because RSI is extreme.\n"
        "- If required velocity to win exceeds trailing volatility, prefer NO_TRADE.\n"
        "- If consecutive directional ticks are 8 or more, do not chase further in that same direction.\n"
        "- If ADX(14) is above 35, do not trade against the trend.\n"
        "- If ADX(14) is above 45, assume the trend may be exhausted and prefer reversal setups over late trend-chasing.\n"
        "- Use EMA alignment and EMA cross direction as directional bias filters.\n"
        "- Use RSI speed divergence to catch short-term exhaustion.\n"
        "- Normalize large target gaps against ATR before taking late-window trades.\n"
        "- If momentum acceleration is moving against the current momentum, treat the move as weakening.\n"
        "- Use velocity_15s and velocity_30s to detect late reversals and falling-knife setups.\n"
        "- If Window Delta < 0.005% near T-10, prefer NO_TRADE.\n"
        "- If Window Delta > 0.15% near T-10, confidence should usually be 0.95 or higher.\n"
        "- If confidence > 0.90, assume no extra edge buffer is required.\n"
        "- If time remaining < 5 seconds and confidence > 0.70, prefer a directional trade over NO_TRADE.\n"
        "- The execution layer will apply regime-aware EV, deadline, liquidity, and FOK rules.\n"
        "- `max_price_to_pay` is ignored by execution; set it to 1.0 for directional trades.\n\n"
        "Keep `reason` short and concrete.\n"
        "Return ONLY the JSON object described in the system message."
    )


def _build_compact_user_prompt(features: BtcFeatures, market: BtcUpDownMarket, up_snapshot=None, down_snapshot=None) -> str:
    time_remaining_seconds = max(market.end_ts - int(features.as_of.timestamp()), 0)
    gap_to_target = (
        None
        if market.settlement_threshold in (None, 0)
        else features.price_usd - market.settlement_threshold
    )
    required_velocity_to_win = (
        None
        if gap_to_target is None or time_remaining_seconds <= 0
        else abs(gap_to_target) / time_remaining_seconds
    )
    return (
        f"BTC 5m market slug: {market.slug}\n"
        f"Price to beat USD: {market.settlement_threshold}\n"
        f"Time remaining seconds: {time_remaining_seconds}\n"
        f"Current BTC price USD: {features.price_usd:.2f}\n"
        f"Window Delta pct: {features.delta_pct_from_window_open * 100:.4f}%\n"
        f"UP ask price: {getattr(up_snapshot, 'buy_quote', None)}\n"
        f"DOWN ask price: {getattr(down_snapshot, 'buy_quote', None)}\n"
        f"UP imbalance: {getattr(up_snapshot, 'top_level_book_imbalance', None)}\n"
        f"DOWN imbalance: {getattr(down_snapshot, 'top_level_book_imbalance', None)}\n"
        f"Req velocity to win: {required_velocity_to_win}\n"
        f"Window open price USD: {features.window_open_price:.2f}\n"
        f"Trailing 5-minute open USD: {features.trailing_5m_open_price:.2f}\n"
        f"Delta from trailing 5-minute open pct: {features.delta_pct_from_trailing_5m_open * 100:.4f}%\n"
        f"Change from previous tick USD: {features.delta_from_previous_tick}\n"
        f"RSI(9): {features.rsi_9}\n"
        f"RSI(14): {features.rsi_14}\n"
        f"RSI speed divergence: {features.rsi_speed_divergence}\n"
        f"1-minute momentum USD: {features.momentum_1m}\n"
        f"Trailing 5-minute momentum USD: {features.momentum_5m}\n"
        f"Velocity 15s USD: {features.velocity_15s}\n"
        f"Velocity 30s USD: {features.velocity_30s}\n"
        f"Momentum acceleration: {features.momentum_acceleration}\n"
        f"EMA9: {features.ema_9}\n"
        f"EMA21: {features.ema_21}\n"
        f"EMA alignment: {features.ema_alignment}\n"
        f"EMA cross: {features.ema_cross_direction}\n"
        f"ADX14: {features.adx_14}\n"
        f"ATR14: {features.atr_14}\n"
        f"Trailing 5-minute volatility: {features.volatility_5m}\n"
        f"Directional ticks: {features.consecutive_directional_ticks}\n"
        "Settlement: UP wins only above the price to beat; DOWN wins only below it.\n"
        "Do not fade parabolic trend and do not chase if directional ticks are >= 8.\n"
        "If ADX14 > 35, do not fight the trend. If ADX14 > 45, avoid late trend-chasing and look for exhaustion/reversal logic.\n"
        "If required velocity to win exceeds volatility, prefer NO_TRADE.\n"
        "Provide direction plus confidence as win probability. Execution handles EV and timing.\n"
        'Return one JSON object with keys: decision, confidence, max_price_to_pay, reason.'
    )


def _build_minimal_user_prompt(features: BtcFeatures, market: BtcUpDownMarket, up_snapshot=None, down_snapshot=None) -> str:
    time_remaining_seconds = max(market.end_ts - int(features.as_of.timestamp()), 0)
    gap_to_target = (
        None
        if market.settlement_threshold in (None, 0)
        else features.price_usd - market.settlement_threshold
    )
    required_velocity_to_win = (
        None
        if gap_to_target is None or time_remaining_seconds <= 0
        else abs(gap_to_target) / time_remaining_seconds
    )
    return (
        f"beat={market.settlement_threshold}\n"
        f"t={time_remaining_seconds}\n"
        f"btc={features.price_usd:.2f}\n"
        f"delta_pct={features.delta_pct_from_window_open * 100:.4f}\n"
        f"up_ask={getattr(up_snapshot, 'buy_quote', None)}\n"
        f"down_ask={getattr(down_snapshot, 'buy_quote', None)}\n"
        f"up_imb={getattr(up_snapshot, 'top_level_book_imbalance', None)}\n"
        f"dn_imb={getattr(down_snapshot, 'top_level_book_imbalance', None)}\n"
        f"rsi9={features.rsi_9}\n"
        f"rsi14={features.rsi_14}\n"
        f"rsi_div={features.rsi_speed_divergence}\n"
        f"mom1m={features.momentum_1m}\n"
        f"mom5m={features.momentum_5m}\n"
        f"v15={features.velocity_15s}\n"
        f"v30={features.velocity_30s}\n"
        f"acc={features.momentum_acceleration}\n"
        f"ema9={features.ema_9}\n"
        f"ema21={features.ema_21}\n"
        f"ema_ok={features.ema_alignment}\n"
        f"adx14={features.adx_14}\n"
        f"atr14={features.atr_14}\n"
        f"vol5m={features.volatility_5m}\n"
        f"reqv={required_velocity_to_win}\n"
        f"dir_ticks={features.consecutive_directional_ticks}\n"
        "UP above beat. DOWN below beat.\n"
        "No fade of parabolic trend; no chase if dir_ticks>=8; if adx14>35 follow trend; if adx14>45 expect exhaustion; if reqv>vol5m prefer NO_TRADE.\n"
        "Return direction + confidence as win probability.\n"
        'Return one JSON object with keys: decision, confidence, max_price_to_pay, reason.'
    )


def _build_openai_realtime_user_prompt(
    features: BtcFeatures,
    market: BtcUpDownMarket,
    up_snapshot=None,
    down_snapshot=None,
) -> str:
    time_remaining_seconds = max(market.end_ts - int(features.as_of.timestamp()), 0)
    gap_to_target = (
        None
        if market.settlement_threshold in (None, 0)
        else features.price_usd - market.settlement_threshold
    )
    required_velocity_to_win = (
        None
        if gap_to_target is None or time_remaining_seconds <= 0
        else abs(gap_to_target) / time_remaining_seconds
    )
    return (
        f"beat={market.settlement_threshold};"
        f"t={time_remaining_seconds};"
        f"btc={features.price_usd:.2f};"
        f"d={features.delta_pct_from_window_open * 100:.4f};"
        f"u={getattr(up_snapshot, 'buy_quote', None)};"
        f"dn={getattr(down_snapshot, 'buy_quote', None)};"
        f"ui={getattr(up_snapshot, 'top_level_book_imbalance', None)};"
        f"di={getattr(down_snapshot, 'top_level_book_imbalance', None)};"
        f"r9={features.rsi_9};"
        f"rsi={features.rsi_14};"
        f"rd={features.rsi_speed_divergence};"
        f"m1={features.momentum_1m};"
        f"m5={features.momentum_5m};"
        f"v15={features.velocity_15s};"
        f"v30={features.velocity_30s};"
        f"acc={features.momentum_acceleration};"
        f"e9={features.ema_9};"
        f"e21={features.ema_21};"
        f"ea={features.ema_alignment};"
        f"adx={features.adx_14};"
        f"atr={features.atr_14};"
        f"v5={features.volatility_5m};"
        f"reqv={required_velocity_to_win};"
        f"dt={features.consecutive_directional_ticks};"
        "json only"
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


def _get_openai_realtime_model(configured_model: str) -> str:
    if configured_model and "realtime" in configured_model:
        return configured_model
    override = os.getenv("OPENAI_REALTIME_MODEL", "").strip()
    if override:
        return override
    return "gpt-realtime-mini"


class OpenAIRealtimeClient:
    def __init__(self, api_key: str, model: str, timeout_seconds: float) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.ws = None
        self._lock = threading.Lock()
        self._request_count = 0

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    def _connect(self) -> None:
        self.close()
        self.ws = websocket.create_connection(
            f"wss://api.openai.com/v1/realtime?model={self.model}",
            header=[
                f"Authorization: Bearer {self.api_key}",
                "OpenAI-Beta: realtime=v1",
            ],
            timeout=self.timeout_seconds,
            enable_multithread=True,
        )
        self.ws.settimeout(self.timeout_seconds)

    def _ensure_connected(self) -> None:
        if self.ws is None:
            self._connect()

    def request(self, system_prompt: str, user_prompt: str) -> str:
        with self._lock:
            self._ensure_connected()
            if self._request_count >= 20:
                self._connect()
                self._request_count = 0
            request_id = str(uuid.uuid4())
            try:
                self.ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "instructions": system_prompt,
                                "modalities": ["text"],
                            },
                        }
                    )
                )
                self.ws.send(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": user_prompt,
                                    }
                                ],
                            },
                        }
                    )
                )
                self.ws.send(
                    json.dumps(
                        {
                            "type": "response.create",
                            "response": {
                                "modalities": ["text"],
                                "max_output_tokens": 160,
                                "metadata": {"request_id": request_id},
                            },
                        }
                    )
                )
                chunks = []
                while True:
                    raw_message = self.ws.recv()
                    event = json.loads(raw_message)
                    event_type = event.get("type")
                    if event_type in {"response.output_text.delta", "response.text.delta"}:
                        delta = event.get("delta") or ""
                        if delta:
                            chunks.append(str(delta))
                    elif event_type in {"response.output_text.done", "response.text.done"}:
                        text = event.get("text") or ""
                        if text and not chunks:
                            chunks.append(str(text))
                    elif event_type == "response.done":
                        break
                    elif event_type == "error":
                        error = event.get("error") or {}
                        raise RuntimeError(str(error.get("message") or event))
                self._request_count += 1
                if not chunks:
                    raise RuntimeError("OpenAI Realtime response contained no content")
                return "".join(chunks)
            except Exception:
                self.close()
                raise


def _get_openai_realtime_client(api_key: str, model: str, timeout_seconds: float) -> OpenAIRealtimeClient:
    global _OPENAI_REALTIME_CLIENT
    realtime_model = _get_openai_realtime_model(model)
    with _OPENAI_REALTIME_CLIENT_LOCK:
        if (
            _OPENAI_REALTIME_CLIENT is None
            or _OPENAI_REALTIME_CLIENT.api_key != api_key
            or _OPENAI_REALTIME_CLIENT.model != realtime_model
        ):
            if _OPENAI_REALTIME_CLIENT is not None:
                _OPENAI_REALTIME_CLIENT.close()
            _OPENAI_REALTIME_CLIENT = OpenAIRealtimeClient(
                api_key=api_key,
                model=realtime_model,
                timeout_seconds=timeout_seconds,
            )
        return _OPENAI_REALTIME_CLIENT


def _stream_openai_chat_completion(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float,
) -> str:
    session = requests.Session()
    session.trust_env = False
    try:
        with session.post(
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
                "stream": True,
            },
            timeout=timeout_seconds,
            stream=True,
        ) as response:
            response.raise_for_status()
            chunks = []
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data_text = line[len("data:") :].strip()
                if data_text == "[DONE]":
                    break
                payload = json.loads(data_text)
                choices = payload.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    chunks.append(str(content))
            if not chunks:
                raise RuntimeError("OpenAI streaming response contained no content")
            return "".join(chunks)
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
    realtime_client = _get_openai_realtime_client(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    return realtime_client.request(system_prompt=system_prompt, user_prompt=user_prompt)


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


def decide_trade(features: BtcFeatures, market: BtcUpDownMarket, up_snapshot=None, down_snapshot=None) -> LlmDecision:
    cfg = get_llm_config()
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(features, market, up_snapshot=up_snapshot, down_snapshot=down_snapshot)
    compact_user_prompt = _build_compact_user_prompt(features, market, up_snapshot=up_snapshot, down_snapshot=down_snapshot)
    minimal_user_prompt = _build_minimal_user_prompt(features, market, up_snapshot=up_snapshot, down_snapshot=down_snapshot)
    openai_system_prompt = _build_openai_realtime_system_prompt()
    openai_user_prompt = _build_openai_realtime_user_prompt(
        features,
        market,
        up_snapshot=up_snapshot,
        down_snapshot=down_snapshot,
    )
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
                system_prompt=openai_system_prompt,
                user_prompt=openai_user_prompt,
                fallback_user_prompt=None,
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
