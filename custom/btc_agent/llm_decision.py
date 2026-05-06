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
from .config import get_llm_config, get_trading_config
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
    prompt_text: Optional[str] = None
    raw_response_text: Optional[str] = None


_OPENAI_REALTIME_CLIENT = None
_OPENAI_REALTIME_CLIENT_LOCK = threading.Lock()


def _slug_start_ts(slug: Optional[str]) -> Optional[int]:
    if not slug:
        return None
    match = re.search(r"btc-updown-5m-(\d+)$", str(slug))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _get_time_remaining_seconds(market: BtcUpDownMarket, as_of_ts: int) -> int:
    slug_start_ts = _slug_start_ts(getattr(market, "slug", None))
    canonical_start_ts = slug_start_ts or getattr(market, "start_ts", None)
    canonical_end_ts = canonical_start_ts + 300 if canonical_start_ts else None
    effective_end_ts = getattr(market, "end_ts", None)
    if canonical_end_ts is not None:
        if effective_end_ts is None or canonical_end_ts > effective_end_ts:
            effective_end_ts = canonical_end_ts
    if effective_end_ts is None:
        return 0
    return max(int(effective_end_ts) - as_of_ts, 0)


def _compute_implied_oracle_price(
    features: BtcFeatures,
    market: BtcUpDownMarket,
    up_snapshot=None,
    down_snapshot=None,
) -> Optional[float]:
    atr_14 = getattr(features, "atr_14", None)
    if (
        atr_14 in (None, 0)
        or market.settlement_threshold in (None, 0)
        or up_snapshot is None
        or down_snapshot is None
        or getattr(up_snapshot, "buy_quote", None) is None
        or getattr(down_snapshot, "buy_quote", None) is None
    ):
        return None
    return (
        float(market.settlement_threshold)
        + (float(up_snapshot.buy_quote) - float(down_snapshot.buy_quote)) * float(atr_14)
    )


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
        "time_remaining_seconds is authoritative. Do not infer time from any other number.\n"
        "Final 10 seconds means time_remaining_seconds < 15.\n"
        "If time_remaining_seconds > 240, you are in the Discovery Phase. Avoid high-confidence trades unless trend intensity is extreme.\n"
        "Use DISTANCE_FROM_STRIKE_PCT to determine whether UP or DOWN is currently winning versus the price to beat. A positive value means BTC is above the strike; a negative value means BTC is below the strike.\n"
        "Do not confuse DISTANCE_FROM_STRIKE_USD or DISTANCE_FROM_STRIKE_PCT with MARKET_WIN_CHANCE_UP / MARKET_WIN_CHANCE_DOWN. Distance fields are price gaps; market win chance fields are market-implied probabilities.\n"
        "Treat Window Delta as a recent-drift confidence signal only in the final 10 seconds.\n"
        "Window Delta means the percent change from the market window open price only. Do not confuse it with DISTANCE_FROM_STRIKE_PCT, DISTANCE_FROM_STRIKE_USD, oracle_gap_ratio, or any ATR-normalized value.\n"
        "window_delta_pct is not the settlement baseline. velocity_30s is micro-momentum for entry timing only, not side selection.\n"
        "If Window Delta is below 0.005% near T-10, ignore TA noise and prefer NO_TRADE.\n"
        "If Window Delta is above 0.15% near T-10, confidence should usually be 0.95 or higher.\n"
        "If confidence is above 0.90, treat it as a directive to get in rather than demanding extra edge buffer.\n"
        "If time remaining is under 5 seconds and confidence is above 0.70, avoid NO_TRADE unless the signal is clearly invalid.\n"
        "Respect market consensus. If the chosen side market win chance is below 0.10 and time_remaining_seconds is greater than 180, prefer NO_TRADE. If the chosen side market win chance is below 0.15 and 15 <= time_remaining_seconds < 120, prefer NO_TRADE. Under 15 seconds, only fade consensus on a clear reversal.\n"
        "If DISTANCE_FROM_STRIKE_PCT is positive and you choose DOWN, confidence must be below 0.50 unless trend exhaustion is clear. If DISTANCE_FROM_STRIKE_PCT is negative and you choose UP, confidence must be below 0.50 unless trend exhaustion is clear.\n"
        "`max_price_to_pay` is informational only and is not used by execution.\n"
        "For directional trades, set `max_price_to_pay` to 1.0 unless you have a strong reason not to.\n"
        "If Window Delta is above 0.15% near T-10, you may set `max_price_to_pay` as high as 0.97."
    )


def _build_openai_realtime_system_prompt() -> str:
    return (
        "Return one JSON object only: decision, confidence, max_price_to_pay, reason. "
        "decision=UP|DOWN|NO_TRADE. confidence is win probability 0..1. "
        "Use DISTANCE_FROM_STRIKE_USD and DISTANCE_FROM_STRIKE_PCT as the settlement baseline: positive means above strike, negative means below strike. "
        "Use MARKET_WIN_CHANCE_UP and MARKET_WIN_CHANCE_DOWN as crowd consensus and velocity_30s only for entry timing, not side selection. "
        "Do not confuse strike-distance fields with market-win-chance fields. "
        "If time_remaining_seconds>240, avoid high-confidence trades unless trend is extreme. "
        "If chosen market win chance<0.10 and time_remaining_seconds>180, prefer NO_TRADE. "
        "If chosen market win chance<0.15 and 15<=time_remaining_seconds<120, prefer NO_TRADE. "
        "If reqv>vol5m/10, prefer NO_TRADE. "
        "If confidence differs from market implied probability by more than 0.50, prefer NO_TRADE. "
        "If rsi9<30, do not choose DOWN. If rsi9>70, do not choose UP. "
        "If DISTANCE_FROM_STRIKE_PCT>0 and choosing DOWN, confidence must stay below 0.50 unless exhaustion is clear; symmetric for UP when DISTANCE_FROM_STRIKE_PCT<0. "
        "Use 1.0 for max_price_to_pay on directional trades."
    )


def _build_user_prompt(features: BtcFeatures, market: BtcUpDownMarket, up_snapshot=None, down_snapshot=None) -> str:
    time_remaining_seconds = _get_time_remaining_seconds(market, int(features.as_of.timestamp()))
    implied_oracle_price = _compute_implied_oracle_price(features, market, up_snapshot, down_snapshot)
    effective_current_price = (
        implied_oracle_price if implied_oracle_price is not None else features.price_usd
    )
    gap_to_target = (
        None
        if market.settlement_threshold in (None, 0)
        else effective_current_price - market.settlement_threshold
    )
    required_velocity_to_win = (
        None
        if gap_to_target is None or time_remaining_seconds <= 0
        else abs(gap_to_target) / time_remaining_seconds
    )
    strike_delta_pct = (
        None
        if gap_to_target is None or features.price_usd in (None, 0)
        else gap_to_target / features.price_usd
    )
    strike_delta_pct_display = (
        "None" if strike_delta_pct is None else f"{strike_delta_pct * 100:.4f}%"
    )
    trend_intensity = features.adx_14
    oracle_gap_ratio = (
        None
        if gap_to_target is None or features.atr_14 in (None, 0)
        else gap_to_target / features.atr_14
    )
    return (
        f"Market title: {market.title}\n"
        f"Market slug: {market.slug}\n\n"
        "Market reference:\n"
        f"- Price to beat USD: {market.settlement_threshold}\n"
        f"- Settlement rule: UP wins only if BTC finishes above {market.settlement_threshold}; "
        f"DOWN wins only if BTC finishes below {market.settlement_threshold}.\n"
        f"- Time remaining seconds: {time_remaining_seconds}\n"
        f"- DISTANCE_FROM_STRIKE_PCT: {strike_delta_pct_display}\n"
        f"- Window Delta pct: {features.delta_pct_from_window_open * 100:.4f}%\n"
        f"- UP Polymarket ask/buy quote: {getattr(up_snapshot, 'buy_quote', None)}\n"
        f"- DOWN Polymarket ask/buy quote: {getattr(down_snapshot, 'buy_quote', None)}\n"
        f"- UP top-book imbalance: {getattr(up_snapshot, 'top_level_book_imbalance', None)}\n"
        f"- DOWN top-book imbalance: {getattr(down_snapshot, 'top_level_book_imbalance', None)}\n"
        f"- UP imbalance pressure: {getattr(up_snapshot, 'imbalance_pressure', None)}\n"
        f"- DOWN imbalance pressure: {getattr(down_snapshot, 'imbalance_pressure', None)}\n"
        f"- Required velocity to win USD/sec: {required_velocity_to_win}\n\n"
        "BTC features:\n"
        f"- Current BTC price USD (raw feed): {features.price_usd:.2f}\n"
        f"- Effective BTC price USD (drift-adjusted): {effective_current_price:.2f}\n"
        f"- DISTANCE_FROM_STRIKE_PCT (BTC vs price to beat): {strike_delta_pct_display}\n"
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
        f"- Trend intensity (ADX): {trend_intensity}\n"
        f"- ATR(14): {features.atr_14}\n"
        f"- Oracle gap ratio: {oracle_gap_ratio}\n"
        f"- Trailing 5-minute volatility: {features.volatility_5m}\n"
        f"- Consecutive flat ticks: {features.consecutive_flat_ticks}\n"
        f"- Consecutive directional ticks: {features.consecutive_directional_ticks}\n"
        f"- Last 10 ticks direction: {features.last_10_ticks_direction}\n\n"
        "Decision policy:\n"
        "- Focus on regime detection and direction, not limit pricing.\n"
        "- Confidence should represent your estimated win probability for the chosen side.\n"
        "- time_remaining_seconds is authoritative. Final 10 seconds means time_remaining_seconds < 15.\n"
        "- If time_remaining_seconds > 240, you are in the Discovery Phase. Avoid high-confidence trades unless trend intensity is extreme.\n"
        "- DISTANCE_FROM_STRIKE_PCT is the source of truth for whether UP or DOWN is currently winning against the price to beat.\n"
        "- A positive DISTANCE_FROM_STRIKE_PCT means BTC is above the strike and UP is currently winning. A negative DISTANCE_FROM_STRIKE_PCT means BTC is below the strike and DOWN is currently winning.\n"
        "- Use the drift-adjusted Effective BTC price as the true current price when reasoning about distance to the strike.\n"
        "- Window Delta is a recent-drift confidence signal near T-10 only.\n"
        "- Window Delta means percent change from market window open only. Do not confuse it with DISTANCE_FROM_STRIKE_PCT, DISTANCE_FROM_STRIKE_USD, MARKET_WIN_CHANCE_UP, MARKET_WIN_CHANCE_DOWN, or Oracle gap ratio.\n"
        "- velocity_30s is micro-momentum for entry timing only; do not use velocity_30s alone to choose UP or DOWN.\n"
        "- Treat order-book imbalance and imbalance pressure as leading indicators.\n"
        "- Do not fade PARABOLIC_UP or PARABOLIC_DOWN regimes just because RSI is extreme.\n"
        "- If the chosen side MARKET_WIN_CHANCE is below 0.10 and time_remaining_seconds is greater than 180, prefer NO_TRADE.\n"
        "- If the chosen side MARKET_WIN_CHANCE is below 0.15 and 15 <= time_remaining_seconds < 120, prefer NO_TRADE.\n"
        "- Under 15 seconds, only bet against a sub-0.15 side quote when velocity_30s and momentum_acceleration show a clear reversal. Apply this symmetrically for UP and DOWN.\n"
        "- If time_remaining_seconds is greater than 60 and abs(gap_to_target_usd) is less than 0.2 * volatility_5m, the market is too close to call and you should prefer NO_TRADE.\n"
        "- If you want UP while the UP buy quote is below 0.45, prefer NO_TRADE because the market is not confirming the breakout.\n"
        "- If RSI(9) is above 85 and BTC is already above the strike, do not choose UP unless time_remaining_seconds is under 15 and continuation is exceptionally clear.\n"
        "- If DISTANCE_FROM_STRIKE_PCT is positive and you choose DOWN, confidence must be below 0.50 unless trend exhaustion is clear. Apply the same rule symmetrically for UP when DISTANCE_FROM_STRIKE_PCT is negative.\n"
        "- If required velocity to win exceeds volatility_5m / 10, prefer NO_TRADE.\n"
        "- If RSI(9) is below 30, do not choose DOWN.\n"
        "- If RSI(9) is above 70, do not choose UP.\n"
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
    time_remaining_seconds = _get_time_remaining_seconds(market, int(features.as_of.timestamp()))
    implied_oracle_price = _compute_implied_oracle_price(features, market, up_snapshot, down_snapshot)
    effective_current_price = (
        implied_oracle_price if implied_oracle_price is not None else features.price_usd
    )
    gap_to_target = (
        None
        if market.settlement_threshold in (None, 0)
        else effective_current_price - market.settlement_threshold
    )
    required_velocity_to_win = (
        None
        if gap_to_target is None or time_remaining_seconds <= 0
        else abs(gap_to_target) / time_remaining_seconds
    )
    strike_delta_pct = (
        None
        if gap_to_target is None or features.price_usd in (None, 0)
        else gap_to_target / features.price_usd
    )
    strike_delta_pct_display = (
        "None" if strike_delta_pct is None else f"{strike_delta_pct * 100:.4f}%"
    )
    oracle_gap_ratio = (
        None
        if gap_to_target is None or features.atr_14 in (None, 0)
        else gap_to_target / features.atr_14
    )
    return (
        f"BTC 5m market slug: {market.slug}\n"
        f"Price to beat USD: {market.settlement_threshold}\n"
        f"Time remaining seconds: {time_remaining_seconds}\n"
        f"Current BTC price USD (raw): {features.price_usd:.2f}\n"
        f"Effective BTC price USD: {effective_current_price:.2f}\n"
        f"DISTANCE_FROM_STRIKE_PCT: {strike_delta_pct_display}\n"
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
        f"Trend intensity: {features.adx_14}\n"
        f"ATR14: {features.atr_14}\n"
        f"Oracle gap ratio: {oracle_gap_ratio}\n"
        f"Trailing 5-minute volatility: {features.volatility_5m}\n"
        f"Directional ticks: {features.consecutive_directional_ticks}\n"
        f"Last 10 ticks direction: {features.last_10_ticks_direction}\n"
        "Settlement: UP wins only above the price to beat; DOWN wins only below it.\n"
        "time_remaining_seconds is authoritative; final 10 seconds means <15, and >240 is Discovery Phase.\n"
        "DISTANCE_FROM_STRIKE_PCT is the source of truth for whether UP or DOWN is currently winning versus the strike.\n"
        "Use Effective BTC price as the true current price when reasoning about the strike gap.\n"
        "Window Delta only means change from market window open, never DISTANCE_FROM_STRIKE_PCT, DISTANCE_FROM_STRIKE_USD, MARKET_WIN_CHANCE fields, or Oracle gap ratio.\n"
        "velocity_30s is for entry timing only.\n"
        "Do not fade parabolic trend and do not chase if directional ticks are >= 8.\n"
        "Do not confuse DISTANCE_FROM_STRIKE fields with MARKET_WIN_CHANCE fields.\n"
        "If chosen MARKET_WIN_CHANCE is below 0.10 and time_remaining_seconds is greater than 180, prefer NO_TRADE.\n"
        "If chosen MARKET_WIN_CHANCE is below 0.15 and 15 <= time_remaining_seconds < 120, prefer NO_TRADE.\n"
        "If time_remaining_seconds > 60 and abs(gap_to_target_usd) < 0.2 * volatility_5m, prefer NO_TRADE.\n"
        "If choosing UP while UP quote < 0.45, prefer NO_TRADE.\n"
        "If RSI(9) > 85 and BTC is already above the strike, do not choose UP unless time_remaining_seconds < 15 and continuation is exceptionally clear.\n"
        "If RSI(9) < 30, do not choose DOWN. If RSI(9) > 70, do not choose UP.\n"
        "If DISTANCE_FROM_STRIKE_PCT is positive and you choose DOWN, confidence must be below 0.50 unless exhaustion is clear; same symmetrically for UP when DISTANCE_FROM_STRIKE_PCT is negative.\n"
        "If ADX14 > 35, do not fight the trend. If ADX14 > 45, avoid late trend-chasing and look for exhaustion/reversal logic.\n"
        "If required velocity to win exceeds volatility_5m / 10, prefer NO_TRADE.\n"
        "Provide direction plus confidence as win probability. Execution handles EV and timing.\n"
        'Return one JSON object with keys: decision, confidence, max_price_to_pay, reason.'
    )


def _build_minimal_user_prompt(features: BtcFeatures, market: BtcUpDownMarket, up_snapshot=None, down_snapshot=None) -> str:
    time_remaining_seconds = _get_time_remaining_seconds(market, int(features.as_of.timestamp()))
    implied_oracle_price = _compute_implied_oracle_price(features, market, up_snapshot, down_snapshot)
    effective_current_price = (
        implied_oracle_price if implied_oracle_price is not None else features.price_usd
    )
    gap_to_target = (
        None
        if market.settlement_threshold in (None, 0)
        else effective_current_price - market.settlement_threshold
    )
    required_velocity_to_win = (
        None
        if gap_to_target is None or time_remaining_seconds <= 0
        else abs(gap_to_target) / time_remaining_seconds
    )
    strike_delta_pct = (
        None
        if gap_to_target is None or features.price_usd in (None, 0)
        else gap_to_target / features.price_usd
    )
    strike_delta_usd = gap_to_target
    strike_delta_pct_display = (
        "None" if strike_delta_pct is None else f"{strike_delta_pct * 100:.4f}"
    )
    return (
        f"beat={market.settlement_threshold}\n"
        f"t={time_remaining_seconds}\n"
        f"btc_raw={features.price_usd:.2f}\n"
        f"btc_eff={effective_current_price:.2f}\n"
        f"DISTANCE_FROM_STRIKE_USD={strike_delta_usd}\n"
        f"DISTANCE_FROM_STRIKE_PCT={strike_delta_pct_display}\n"
        f"MARKET_WIN_CHANCE_UP={market.up_market_probability}\n"
        f"MARKET_WIN_CHANCE_DOWN={market.down_market_probability}\n"
        f"up_ask={getattr(up_snapshot, 'buy_quote', None)}\n"
        f"down_ask={getattr(down_snapshot, 'buy_quote', None)}\n"
        f"rsi9={features.rsi_9}\n"
        f"mom1m={features.momentum_1m}\n"
        f"v30={features.velocity_30s}\n"
        f"acc={features.momentum_acceleration}\n"
        f"adx14={features.adx_14}\n"
        f"vol5m={features.volatility_5m}\n"
        f"reqv={required_velocity_to_win}\n"
        f"dir_ticks={features.consecutive_directional_ticks}\n"
        "UP above beat. DOWN below beat.\n"
        "t is authoritative; final 10 seconds means t<15; if t>240 you are in Discovery Phase.\n"
        "DISTANCE_FROM_STRIKE_USD and DISTANCE_FROM_STRIKE_PCT are the settlement baseline; positive means above strike, negative means below strike.\n"
        "MARKET_WIN_CHANCE_UP and MARKET_WIN_CHANCE_DOWN are market-implied probabilities, not price distances.\n"
        "Do not confuse DISTANCE_FROM_STRIKE values with MARKET_WIN_CHANCE values.\n"
        "Use btc_eff as the true current price for strike-gap reasoning.\n"
        "Ignore window-open drift. v30 is entry timing only.\n"
        "MARKET_WIN_CHANCE_UP and MARKET_WIN_CHANCE_DOWN come from Gamma. Do not bet against them lightly.\n"
        "No fade of parabolic trend; no chase if dir_ticks>=8; if adx14>35 follow trend; if adx14>45 expect exhaustion; if reqv>(vol5m/10) prefer NO_TRADE.\n"
        "If chosen side MARKET_WIN_CHANCE <0.10 and t>180, prefer NO_TRADE.\n"
        "If chosen side MARKET_WIN_CHANCE <0.15 and 15<=t<120, prefer NO_TRADE.\n"
        "If t>60 and abs(btc-beat) < 0.2*vol5m, prefer NO_TRADE.\n"
        "If choosing UP and up_ask<0.45, prefer NO_TRADE.\n"
        "If rsi9>85 and btc>beat, do not choose UP unless t<15 and continuation is exceptionally clear.\n"
        "If rsi9<30, do not choose DOWN. If rsi9>70, do not choose UP.\n"
        "If DISTANCE_FROM_STRIKE_PCT>0 and choosing DOWN, confidence must stay below 0.50 unless exhaustion is clear; same symmetrically for UP when DISTANCE_FROM_STRIKE_PCT<0.\n"
        "Return direction + confidence as win probability.\n"
        'Return one JSON object with keys: decision, confidence, max_price_to_pay, reason.'
    )


def _build_openai_realtime_user_prompt(
    features: BtcFeatures,
    market: BtcUpDownMarket,
    up_snapshot=None,
    down_snapshot=None,
) -> str:
    time_remaining_seconds = _get_time_remaining_seconds(market, int(features.as_of.timestamp()))
    implied_oracle_price = _compute_implied_oracle_price(features, market, up_snapshot, down_snapshot)
    effective_current_price = (
        implied_oracle_price if implied_oracle_price is not None else features.price_usd
    )
    gap_to_target = (
        None
        if market.settlement_threshold in (None, 0)
        else effective_current_price - market.settlement_threshold
    )
    required_velocity_to_win = (
        None
        if gap_to_target is None or time_remaining_seconds <= 0
        else abs(gap_to_target) / time_remaining_seconds
    )
    strike_delta_pct = (
        None
        if gap_to_target is None or features.price_usd in (None, 0)
        else gap_to_target / features.price_usd
    )
    strike_delta_usd = gap_to_target
    strike_delta_pct_display = (
        "None" if strike_delta_pct is None else f"{strike_delta_pct * 100:.4f}"
    )
    return (
        f"beat={market.settlement_threshold};"
        f"t={time_remaining_seconds};"
        f"btc_eff={effective_current_price:.2f};"
        f"DISTANCE_FROM_STRIKE_USD={strike_delta_usd};"
        f"DISTANCE_FROM_STRIKE_PCT={strike_delta_pct_display};"
        f"MARKET_WIN_CHANCE_UP={market.up_market_probability};"
        f"MARKET_WIN_CHANCE_DOWN={market.down_market_probability};"
        f"u={getattr(up_snapshot, 'buy_quote', None)};"
        f"dn={getattr(down_snapshot, 'buy_quote', None)};"
        f"r9={features.rsi_9};"
        f"m1={features.momentum_1m};"
        f"v30={features.velocity_30s};"
        f"acc={features.momentum_acceleration};"
        f"adx={features.adx_14};"
        f"v5={features.volatility_5m};"
        f"reqv={required_velocity_to_win};"
        f"dt={features.consecutive_directional_ticks};"
        "t_is_authoritative;"
        "if_t_gt_240_discovery_phase;"
        "DISTANCE_FROM_STRIKE_fields_are_settlement_baseline_positive_means_above_strike_negative_means_below_strike;"
        "MARKET_WIN_CHANCE_fields_are_market_probabilities_not_price_distance;"
        "do_not_confuse_distance_from_strike_with_market_win_chance;"
        "btc_eff_is_true_current_price_for_strike_gap;"
        "ignore_window_open_drift_v30_is_entry_timing_only;"
        "MARKET_WIN_CHANCE_UP_and_MARKET_WIN_CHANCE_DOWN_are_gamma_market_probabilities;"
        "if_chosen_market_win_chance_lt_0.10_and_t_gt_180_prefer_no_trade;"
        "if_chosen_market_win_chance_lt_0.15_and_15_lte_t_lt_120_prefer_no_trade;"
        "if_reqv_gt_vol5m_div_10_prefer_no_trade;"
        "if_t_gt_60_and_abs_btc_minus_beat_lt_0.2_vol5m_prefer_no_trade;"
        "if_choose_up_and_u_lt_0.45_prefer_no_trade;"
        "if_r9_gt_85_and_btc_gt_beat_no_up_unless_t_lt_15;"
        "if_r9_lt_30_no_down_if_r9_gt_70_no_up;"
        "if_DISTANCE_FROM_STRIKE_PCT_positive_and_choose_down_confidence_lt_0.50_unless_exhaustion;"
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

    key_value_match = re.search(
        r"decision\s*:\s*(?P<decision>UP|DOWN|NO_TRADE)\s*,\s*"
        r"confidence\s*:\s*(?P<confidence>-?\d+(?:\.\d+)?)\s*,\s*"
        r"max_price_to_pay\s*:\s*(?P<max_price>-?\d+(?:\.\d+)?)\s*,\s*"
        r"reason\s*:\s*(?P<reason>.+)$",
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )
    if key_value_match:
        return {
            "decision": key_value_match.group("decision").upper(),
            "confidence": float(key_value_match.group("confidence")),
            "max_price_to_pay": float(key_value_match.group("max_price")),
            "reason": key_value_match.group("reason").strip(),
        }

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
                                "max_output_tokens": 120,
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
            "maxOutputTokens": 192,
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
) -> tuple[dict, str]:
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
            return _extract_json_payload(raw_text), raw_text
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


def _build_debug_prompt_text(system_prompt: str, user_prompt: str) -> Optional[str]:
    try:
        cfg = get_trading_config()
    except Exception:
        return None
    if not getattr(cfg, "debug", False):
        return None
    return f"SYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_prompt}"


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
        raw_response_text = None
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
    system_prompt = _build_openai_realtime_system_prompt()
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
    debug_prompt_text = (
        _build_debug_prompt_text(openai_system_prompt, openai_user_prompt)
        if cfg.engine == "openai"
        else _build_debug_prompt_text(system_prompt, minimal_user_prompt)
    )

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
            raw_response_text = raw_text
            data = _extract_json_payload(raw_text)
        elif cfg.engine == "gemini":
            data, raw_response_text = _request_gemini_decision_with_parse_retry(
                model=cfg.model,
                api_key=cfg.api_key,
                system_prompt=system_prompt,
                user_prompt=minimal_user_prompt,
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
            prompt_text=debug_prompt_text,
            raw_response_text=None,
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
        prompt_text=debug_prompt_text,
        raw_response_text=raw_response_text,
    )
