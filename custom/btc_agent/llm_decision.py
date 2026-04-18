# custom/btc_agent/llm_decision.py

import json
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI  # v1 SDK

from .config import get_openai_config
from .indicators import BtcFeatures
from .market_lookup import BtcUpDownMarket

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
        '  \"decision\": \"UP\" | \"DOWN\" | \"NO_TRADE\",\n'
        "  \"confidence\": number between 0 and 1,\n"
        "  \"max_price_to_pay\": number between 0 and 1,\n"
        "  \"reason\": string\n"
        "}\n"
        "Be conservative and prefer \"NO_TRADE\" when signals are weak."
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

def decide_trade(features: BtcFeatures, market: BtcUpDownMarket) -> LlmDecision:
    cfg = get_openai_config()
    client = OpenAI(api_key=cfg.api_key)

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(features, market)

    # v1 SDK chat completions, with JSON response_format
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=256,
        response_format={"type": "json_object"},
    )

    raw_text = resp.choices[0].message.content

    data = json.loads(raw_text)

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
