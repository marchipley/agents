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


def _strike_distance_context(
    features: BtcFeatures,
    market: BtcUpDownMarket,
    up_snapshot=None,
    down_snapshot=None,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    implied_oracle_price = _compute_implied_oracle_price(features, market, up_snapshot, down_snapshot)
    current_price = None if features.price_usd is None else float(features.price_usd)
    strike = None if market.settlement_threshold in (None, 0) else float(market.settlement_threshold)
    gap_to_target = None if current_price is None or strike is None else current_price - strike
    strike_delta_pct = None if gap_to_target is None or strike in (None, 0) else gap_to_target / strike
    feed_drift_usd = (
        None
        if current_price is None or implied_oracle_price is None
        else current_price - implied_oracle_price
    )
    return current_price, gap_to_target, strike_delta_pct, implied_oracle_price, feed_drift_usd


def _momentum_alignment_text(features: BtcFeatures) -> str:
    values = [
        getattr(features, "velocity_15s", None),
        getattr(features, "velocity_30s", None),
        getattr(features, "momentum_1m", None),
    ]
    if any(value is None for value in values):
        return "None"
    signs = []
    for value in values:
        if value > 0:
            signs.append(1)
        elif value < 0:
            signs.append(-1)
        else:
            return "False"
    return "True" if signs[0] == signs[1] == signs[2] else "False"


def _build_system_prompt() -> str:
    cfg = get_trading_config()
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
        "Interpret confidence as the direct mathematical probability that your chosen side wins. 1.0 means near-certainty and 0.5 means a coin flip.\n"
        "time_remaining_seconds is authoritative. Do not infer time from any other number.\n"
        "Final 10 seconds means time_remaining_seconds < 15.\n"
        f"If time_remaining_seconds > 240, you are in the Discovery Phase. If ADX < {cfg.discovery_adx_caution_threshold:.0f}, stay cautious. If ADX > {cfg.trend_priority_adx_threshold:.0f}, prioritize the trend over the time elapsed.\n"
        "Use DISTANCE_FROM_STRIKE_PCT to determine whether UP or DOWN is currently winning versus the price to beat. A positive value means BTC is above the strike; a negative value means BTC is below the strike.\n"
        "Do not confuse DISTANCE_FROM_STRIKE_USD or DISTANCE_FROM_STRIKE_PCT with MARKET_WIN_CHANCE_UP / MARKET_WIN_CHANCE_DOWN. Distance fields are price gaps; market win chance fields are market-implied probabilities.\n"
        "A MARKET_WIN_CHANCE of 65% is an aggressive signal. DO NOT confuse this with DISTANCE_FROM_STRIKE. A distance of $1.00 USD is sufficient for 95% confidence if time remaining is less than 15 seconds.\n"
        "Treat Window Delta as a recent-drift confidence signal only in the final 10 seconds.\n"
        "Window Delta means the percent change from the market window open price only. Do not confuse it with DISTANCE_FROM_STRIKE_PCT, DISTANCE_FROM_STRIKE_USD, oracle_gap_ratio, or any ATR-normalized value.\n"
        "window_delta_pct is not the settlement baseline. velocity_30s is micro-momentum for entry timing only, not side selection.\n"
        "If Window Delta is below 0.005% near T-10, ignore TA noise and prefer NO_TRADE.\n"
        "If Window Delta is above 0.15% near T-10, confidence should usually be 0.95 or higher.\n"
        "If confidence is above 0.90, treat it as a directive to get in rather than demanding extra edge buffer.\n"
        "If time remaining is under 5 seconds and confidence is above 0.70, avoid NO_TRADE unless the signal is clearly invalid.\n"
        "CRITICAL EXECUTION RULE: If currently ITM and time remaining < 60s, you may ignore RSI signals ONLY IF the entry price is below 0.80. If the buy quote is > 0.85, the risk-to-reward is too poor to enter a new position regardless of ITM status.\n"
        "Paradoxical Momentum rule: If ADX is above 50 and RSI is above 75, treat this as Exhaustion rather than Strength. Do not enter new positions; only maintain existing ones.\n"
        f"Respect market consensus. If the chosen side market win chance is below {cfg.market_win_chance_veto_threshold:.2f} and 15 <= time_remaining_seconds < {cfg.market_win_chance_veto_end_seconds}, prefer NO_TRADE. Under 15 seconds, only fade consensus on a clear reversal.\n"
        "If momentum_alignment is TRUE, clarity is HIGH and you are encouraged to trade with the trend.\n"
        f"If DISTANCE_FROM_STRIKE_USD is beyond 0.5 * ATR in the chosen direction, you may raise confidence by {cfg.itm_confidence_boost_amount:.2f}.\n"
        "If time_remaining_seconds > 180 and DISTANCE_FROM_STRIKE_USD is less than 0.5 * current 5-minute volatility, prefer NO_TRADE.\n"
        "If time_remaining_seconds > 240 and DISTANCE_FROM_STRIKE_USD is less than 0.2 * ATR, prefer NO_TRADE unless momentum alignment is strong and trend intensity is at maximum.\n"
        "If RSI speed divergence is negative while price is still moving up, treat the move as weakening and lower confidence.\n"
        "If DISTANCE_FROM_STRIKE_PCT is positive and you choose DOWN, confidence must be below 0.50 unless trend exhaustion is clear. If DISTANCE_FROM_STRIKE_PCT is negative and you choose UP, confidence must be below 0.50 unless trend exhaustion is clear.\n"
        "`max_price_to_pay` is informational only and is not used by execution.\n"
        "For directional trades, set `max_price_to_pay` to 1.0 unless you have a strong reason not to.\n"
        "If Window Delta is above 0.15% near T-10, you may set `max_price_to_pay` as high as 0.97."
    )


def _build_openai_realtime_system_prompt() -> str:
    cfg = get_trading_config()
    return (
        "Return one JSON object only: decision, confidence, max_price_to_pay, reason. "
        "decision=UP|DOWN|NO_TRADE. confidence is win probability 0..1. "
        "Use DISTANCE_FROM_STRIKE_USD and DISTANCE_FROM_STRIKE_PCT as the settlement baseline: positive means above strike, negative means below strike. "
        "Use MARKET_WIN_CHANCE_UP and MARKET_WIN_CHANCE_DOWN as crowd consensus and velocity_30s only for entry timing, not side selection. "
        "Do not confuse strike-distance fields with market-win-chance fields. "
        "A MARKET_WIN_CHANCE of 65% is aggressive consensus, not physical distance. A $1.00 DISTANCE_FROM_STRIKE_USD can justify ~95% confidence when time_remaining_seconds<15. "
        f"If time_remaining_seconds>240 and adx<{cfg.discovery_adx_caution_threshold:.0f}, stay cautious; if adx>{cfg.trend_priority_adx_threshold:.0f}, prioritize the trend over elapsed time. "
        "If trade is ITM and time_remaining_seconds<60, you may ignore RSI only when entry quote<0.80; if buy quote>0.85, prefer NO_TRADE regardless of ITM status. "
        "If adx>50 and rsi>75, treat that as exhaustion rather than strength and avoid new entries. "
        f"If chosen market win chance<{cfg.market_win_chance_veto_threshold:.2f} and 15<=time_remaining_seconds<{cfg.market_win_chance_veto_end_seconds}, prefer NO_TRADE. "
        "If momentum_alignment is TRUE, clarity is HIGH and you are encouraged to trade with the trend. "
        f"If DISTANCE_FROM_STRIKE_USD is beyond 0.5 * ATR in the chosen direction, you may raise confidence by {cfg.itm_confidence_boost_amount:.2f}. "
        "If time_remaining_seconds>180 and DISTANCE_FROM_STRIKE_USD<0.5*vol5m, prefer NO_TRADE. "
        "If time_remaining_seconds>240 and abs(DISTANCE_FROM_STRIKE_USD)<0.2*ATR, prefer NO_TRADE unless momentum_alignment is TRUE and ADX is at maximum trend strength. "
        f"If chosen side is OTM and reqv>vol5m/{cfg.required_velocity_divisor:.0f}, prefer NO_TRADE; if chosen side is ITM, ignore reqv. "
        "If confidence differs from market implied probability by more than 0.50, prefer NO_TRADE. "
        f"If rsi9<{cfg.down_rsi_veto_threshold:.0f}, do not choose DOWN. If rsi9>{cfg.up_rsi_veto_base_threshold:.0f}, do not choose UP unless adx>{cfg.up_rsi_veto_adx_threshold:.0f} and continuation remains strong up to rsi9>{cfg.up_rsi_veto_trend_threshold:.0f}. "
        "If RSI speed divergence is negative while price is moving up, lower confidence and treat the move as weakening. "
        "If DISTANCE_FROM_STRIKE_PCT>0 and choosing DOWN, confidence must stay below 0.50 unless exhaustion is clear; symmetric for UP when DISTANCE_FROM_STRIKE_PCT<0. "
        "Use 1.0 for max_price_to_pay on directional trades."
    )


def _build_user_prompt(features: BtcFeatures, market: BtcUpDownMarket, up_snapshot=None, down_snapshot=None) -> str:
    cfg = get_trading_config()
    time_remaining_seconds = _get_time_remaining_seconds(market, int(features.as_of.timestamp()))
    (
        effective_current_price,
        gap_to_target,
        strike_delta_pct,
        implied_oracle_price,
        feed_drift_usd,
    ) = _strike_distance_context(features, market, up_snapshot, down_snapshot)
    required_velocity_to_win = (
        None
        if gap_to_target is None or time_remaining_seconds <= 0
        else abs(gap_to_target) / time_remaining_seconds
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
        f"- Effective BTC price USD (raw strike baseline): {effective_current_price:.2f}\n"
        f"- Implied oracle price USD (market context only): {implied_oracle_price}\n"
        f"- Feed drift USD (raw BTC - implied oracle): {feed_drift_usd}\n"
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
        f"- Momentum alignment: {_momentum_alignment_text(features)}\n"
        f"- ATR(14): {features.atr_14}\n"
        f"- Oracle gap ratio: {oracle_gap_ratio}\n"
        f"- Trailing 5-minute volatility: {features.volatility_5m}\n"
        f"- Consecutive flat ticks: {features.consecutive_flat_ticks}\n"
        f"- Consecutive directional ticks: {features.consecutive_directional_ticks}\n"
        f"- Last 10 ticks direction: {features.last_10_ticks_direction}\n\n"
        "Decision policy:\n"
        "- Focus on regime detection and direction, not limit pricing.\n"
        "- Confidence should represent your direct estimated win probability for the chosen side, where 1.0 is near-certainty and 0.5 is a coin flip.\n"
        "- time_remaining_seconds is authoritative. Final 10 seconds means time_remaining_seconds < 15.\n"
        f"- If time_remaining_seconds > 240, you are in the Discovery Phase. If ADX < {cfg.discovery_adx_caution_threshold:.0f}, stay cautious. If ADX > {cfg.trend_priority_adx_threshold:.0f}, prioritize the trend over time elapsed.\n"
        "- DISTANCE_FROM_STRIKE_PCT is the source of truth for whether UP or DOWN is currently winning against the price to beat.\n"
        "- A positive DISTANCE_FROM_STRIKE_PCT means BTC is above the strike and UP is currently winning. A negative DISTANCE_FROM_STRIKE_PCT means BTC is below the strike and DOWN is currently winning.\n"
        "- Use raw BTC price / Effective BTC price as the true current price when reasoning about distance to the strike; implied oracle price is market context only.\n"
        "- A MARKET_WIN_CHANCE of 65% is already aggressive consensus. Do not confuse that with physical strike distance. A DISTANCE_FROM_STRIKE_USD of about $1.00 can still justify ~95% confidence if time_remaining_seconds < 15.\n"
        "- Window Delta is a recent-drift confidence signal near T-10 only.\n"
        "- Window Delta means percent change from market window open only. Do not confuse it with DISTANCE_FROM_STRIKE_PCT, DISTANCE_FROM_STRIKE_USD, MARKET_WIN_CHANCE_UP, MARKET_WIN_CHANCE_DOWN, or Oracle gap ratio.\n"
        "- velocity_30s is micro-momentum for entry timing only; do not use velocity_30s alone to choose UP or DOWN.\n"
        "- Treat order-book imbalance and imbalance pressure as leading indicators.\n"
        "- Do not fade PARABOLIC_UP or PARABOLIC_DOWN regimes just because RSI is extreme.\n"
        "- Paradoxical Momentum rule: If ADX is above 50 and RSI is above 75, treat this as Exhaustion rather than Strength. Do not enter new positions; only maintain existing ones.\n"
        f"- If the chosen side MARKET_WIN_CHANCE is below {cfg.market_win_chance_veto_threshold:.2f} and 15 <= time_remaining_seconds < {cfg.market_win_chance_veto_end_seconds}, prefer NO_TRADE.\n"
        "- If momentum alignment is TRUE, clarity is HIGH and you are encouraged to trade with the trend.\n"
        f"- If DISTANCE_FROM_STRIKE_USD is beyond 0.5 * ATR in the chosen direction, you may raise confidence by {cfg.itm_confidence_boost_amount:.2f}.\n"
        "- If time_remaining_seconds > 180 and DISTANCE_FROM_STRIKE_USD is less than 0.5 * current 5-minute volatility, prefer NO_TRADE.\n"
        "- If time_remaining_seconds > 240 and DISTANCE_FROM_STRIKE_USD is less than 0.2 * ATR, prefer NO_TRADE unless momentum alignment is TRUE and ADX is at maximum trend strength.\n"
        f"- Under 15 seconds, only bet against a sub-{cfg.market_win_chance_veto_threshold:.2f} side quote when velocity_30s and momentum_acceleration show a clear reversal. Apply this symmetrically for UP and DOWN.\n"
        "- If time_remaining_seconds is greater than 60 and abs(gap_to_target_usd) is less than 0.2 * volatility_5m, the market is too close to call and you should prefer NO_TRADE.\n"
        "- If you want UP while the UP buy quote is below 0.45, prefer NO_TRADE because the market is not confirming the breakout.\n"
        f"- If RSI(9) is above {cfg.up_rsi_veto_trend_threshold:.0f} and BTC is already above the strike, do not choose UP unless time_remaining_seconds is under 15 and continuation is exceptionally clear.\n"
        "- If DISTANCE_FROM_STRIKE_PCT is positive and you choose DOWN, confidence must be below 0.50 unless trend exhaustion is clear. Apply the same rule symmetrically for UP when DISTANCE_FROM_STRIKE_PCT is negative.\n"
        f"- If the chosen side is OTM and required velocity to win exceeds volatility_5m / {cfg.required_velocity_divisor:.0f}, prefer NO_TRADE. If the chosen side is ITM, ignore required velocity entirely.\n"
        f"- If RSI(9) is below {cfg.down_rsi_veto_threshold:.0f}, do not choose DOWN.\n"
        f"- If RSI(9) is above {cfg.up_rsi_veto_base_threshold:.0f}, do not choose UP unless ADX is above {cfg.up_rsi_veto_adx_threshold:.0f}; even then, treat RSI above {cfg.up_rsi_veto_trend_threshold:.0f} as exhaustion risk.\n"
        "- If consecutive directional ticks are 8 or more, do not chase further in that same direction.\n"
        "- If ADX(14) is above 35, do not trade against the trend.\n"
        "- If ADX(14) is above 45, assume the trend may be exhausted and prefer reversal setups over late trend-chasing.\n"
        "- Use EMA alignment and EMA cross direction as directional bias filters.\n"
        "- Use RSI speed divergence to catch short-term exhaustion.\n"
        "- If RSI speed divergence is negative while price is still moving up, treat the move as weakening and lower confidence.\n"
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
    cfg = get_trading_config()
    time_remaining_seconds = _get_time_remaining_seconds(market, int(features.as_of.timestamp()))
    (
        effective_current_price,
        gap_to_target,
        strike_delta_pct,
        implied_oracle_price,
        feed_drift_usd,
    ) = _strike_distance_context(features, market, up_snapshot, down_snapshot)
    required_velocity_to_win = (
        None
        if gap_to_target is None or time_remaining_seconds <= 0
        else abs(gap_to_target) / time_remaining_seconds
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
        f"Implied oracle price USD (market context only): {implied_oracle_price}\n"
        f"Feed drift USD: {feed_drift_usd}\n"
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
        f"Momentum alignment: {_momentum_alignment_text(features)}\n"
        f"ATR14: {features.atr_14}\n"
        f"Oracle gap ratio: {oracle_gap_ratio}\n"
        f"Trailing 5-minute volatility: {features.volatility_5m}\n"
        f"Directional ticks: {features.consecutive_directional_ticks}\n"
        f"Last 10 ticks direction: {features.last_10_ticks_direction}\n"
        "Settlement: UP wins only above the price to beat; DOWN wins only below it.\n"
        f"time_remaining_seconds is authoritative; final 10 seconds means <15; if ADX < {cfg.discovery_adx_caution_threshold:.0f} stay cautious in Discovery Phase, if ADX > {cfg.trend_priority_adx_threshold:.0f} prioritize trend over elapsed time.\n"
        "DISTANCE_FROM_STRIKE_PCT is the source of truth for whether UP or DOWN is currently winning versus the strike.\n"
        "Use Effective BTC price/raw BTC as the true current price when reasoning about the strike gap; implied oracle price is context only.\n"
        "Window Delta only means change from market window open, never DISTANCE_FROM_STRIKE_PCT, DISTANCE_FROM_STRIKE_USD, MARKET_WIN_CHANCE fields, or Oracle gap ratio.\n"
        "velocity_30s is for entry timing only.\n"
        "Do not fade parabolic trend and do not chase if directional ticks are >= 8.\n"
        "Do not confuse DISTANCE_FROM_STRIKE fields with MARKET_WIN_CHANCE fields.\n"
        f"If chosen MARKET_WIN_CHANCE is below {cfg.market_win_chance_veto_threshold:.2f} and 15 <= time_remaining_seconds < {cfg.market_win_chance_veto_end_seconds}, prefer NO_TRADE.\n"
        "If momentum alignment is TRUE, clarity is HIGH and you are encouraged to trade with the trend.\n"
        f"If DISTANCE_FROM_STRIKE_USD is beyond 0.5 * ATR in the chosen direction, you may raise confidence by {cfg.itm_confidence_boost_amount:.2f}.\n"
        "If time_remaining_seconds > 240 and DISTANCE_FROM_STRIKE_USD is less than 0.2 * ATR, prefer NO_TRADE unless momentum alignment is TRUE and ADX is at maximum trend strength.\n"
        "If time_remaining_seconds > 60 and abs(gap_to_target_usd) < 0.2 * volatility_5m, prefer NO_TRADE.\n"
        "If choosing UP while UP quote < 0.45, prefer NO_TRADE.\n"
        f"If RSI(9) > {cfg.up_rsi_veto_trend_threshold:.0f} and BTC is already above the strike, do not choose UP unless time_remaining_seconds < 15 and continuation is exceptionally clear.\n"
        f"If RSI(9) < {cfg.down_rsi_veto_threshold:.0f}, do not choose DOWN. If RSI(9) > {cfg.up_rsi_veto_base_threshold:.0f}, do not choose UP unless ADX > {cfg.up_rsi_veto_adx_threshold:.0f}.\n"
        "If DISTANCE_FROM_STRIKE_PCT is positive and you choose DOWN, confidence must be below 0.50 unless exhaustion is clear; same symmetrically for UP when DISTANCE_FROM_STRIKE_PCT is negative.\n"
        "If ADX14 > 35, do not fight the trend. If ADX14 > 45, avoid late trend-chasing and look for exhaustion/reversal logic.\n"
        f"If the chosen side is OTM and required velocity to win exceeds volatility_5m / {cfg.required_velocity_divisor:.0f}, prefer NO_TRADE. If the chosen side is ITM, ignore required velocity.\n"
        "If RSI speed divergence is negative while price is still moving up, treat the move as weakening and lower confidence.\n"
        "Provide direction plus confidence as win probability. Execution handles EV and timing.\n"
        'Return one JSON object with keys: decision, confidence, max_price_to_pay, reason.'
    )


def _build_minimal_user_prompt(features: BtcFeatures, market: BtcUpDownMarket, up_snapshot=None, down_snapshot=None) -> str:
    cfg = get_trading_config()
    time_remaining_seconds = _get_time_remaining_seconds(market, int(features.as_of.timestamp()))
    (
        effective_current_price,
        gap_to_target,
        strike_delta_pct,
        implied_oracle_price,
        feed_drift_usd,
    ) = _strike_distance_context(features, market, up_snapshot, down_snapshot)
    required_velocity_to_win = (
        None
        if gap_to_target is None or time_remaining_seconds <= 0
        else abs(gap_to_target) / time_remaining_seconds
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
        f"implied_oracle_price={implied_oracle_price}\n"
        f"feed_drift_usd={feed_drift_usd}\n"
        f"MARKET_WIN_CHANCE_UP={market.up_market_probability}\n"
        f"MARKET_WIN_CHANCE_DOWN={market.down_market_probability}\n"
        f"up_ask={getattr(up_snapshot, 'buy_quote', None)}\n"
        f"down_ask={getattr(down_snapshot, 'buy_quote', None)}\n"
        f"rsi9={features.rsi_9}\n"
        f"rsi_speed_divergence={features.rsi_speed_divergence}\n"
        f"mom1m={features.momentum_1m}\n"
        f"v30={features.velocity_30s}\n"
        f"acc={features.momentum_acceleration}\n"
        f"adx14={features.adx_14}\n"
        f"vol5m={features.volatility_5m}\n"
        f"reqv={required_velocity_to_win}\n"
        f"dir_ticks={features.consecutive_directional_ticks}\n"
        f"momentum_alignment={_momentum_alignment_text(features)}\n"
        "UP above beat. DOWN below beat.\n"
        f"t is authoritative; final 10 seconds means t<15; if t>240 and adx14<{cfg.discovery_adx_caution_threshold:.0f} stay cautious, if adx14>{cfg.trend_priority_adx_threshold:.0f} prioritize trend over elapsed time.\n"
        "DISTANCE_FROM_STRIKE_USD and DISTANCE_FROM_STRIKE_PCT are the settlement baseline; positive means above strike, negative means below strike.\n"
        "MARKET_WIN_CHANCE_UP and MARKET_WIN_CHANCE_DOWN are market-implied probabilities, not price distances.\n"
        "Do not confuse DISTANCE_FROM_STRIKE values with MARKET_WIN_CHANCE values.\n"
        "Use btc_eff/raw BTC as the true current price for strike-gap reasoning; implied_oracle_price is market context only.\n"
        "Ignore window-open drift. v30 is entry timing only.\n"
        "MARKET_WIN_CHANCE_UP and MARKET_WIN_CHANCE_DOWN come from Gamma. Do not bet against them lightly.\n"
        f"No fade of parabolic trend; no chase if dir_ticks>=8; if adx14>35 follow trend; if adx14>45 expect exhaustion; if chosen side is OTM and reqv>(vol5m/{cfg.required_velocity_divisor:.0f}) prefer NO_TRADE, if ITM ignore reqv.\n"
        f"If chosen side MARKET_WIN_CHANCE <{cfg.market_win_chance_veto_threshold:.2f} and 15<=t<{cfg.market_win_chance_veto_end_seconds}, prefer NO_TRADE.\n"
        "If momentum_alignment is TRUE, clarity is HIGH and you are encouraged to trade with the trend.\n"
        "If t>240 and abs(DISTANCE_FROM_STRIKE_USD)<0.2*ATR, prefer NO_TRADE unless momentum_alignment is TRUE and ADX is at maximum trend strength.\n"
        f"If DISTANCE_FROM_STRIKE_USD is beyond 0.5 * ATR in the chosen direction, you may raise confidence by {cfg.itm_confidence_boost_amount:.2f}.\n"
        "If t>60 and abs(btc-beat) < 0.2*vol5m, prefer NO_TRADE.\n"
        "If choosing UP and up_ask<0.45, prefer NO_TRADE.\n"
        f"If rsi9>{cfg.up_rsi_veto_trend_threshold:.0f} and btc>beat, do not choose UP unless t<15 and continuation is exceptionally clear.\n"
        f"If rsi9<{cfg.down_rsi_veto_threshold:.0f}, do not choose DOWN. If rsi9>{cfg.up_rsi_veto_base_threshold:.0f}, do not choose UP unless adx14>{cfg.up_rsi_veto_adx_threshold:.0f}.\n"
        "If rsi_speed_divergence is negative while btc is still moving up, lower confidence and treat the move as weakening.\n"
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
    cfg = get_trading_config()
    time_remaining_seconds = _get_time_remaining_seconds(market, int(features.as_of.timestamp()))
    (
        effective_current_price,
        gap_to_target,
        strike_delta_pct,
        implied_oracle_price,
        feed_drift_usd,
    ) = _strike_distance_context(features, market, up_snapshot, down_snapshot)
    required_velocity_to_win = (
        None
        if gap_to_target is None or time_remaining_seconds <= 0
        else abs(gap_to_target) / time_remaining_seconds
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
        f"implied_oracle_price={implied_oracle_price};"
        f"feed_drift_usd={feed_drift_usd};"
        f"MARKET_WIN_CHANCE_UP={market.up_market_probability};"
        f"MARKET_WIN_CHANCE_DOWN={market.down_market_probability};"
        f"u={getattr(up_snapshot, 'buy_quote', None)};"
        f"dn={getattr(down_snapshot, 'buy_quote', None)};"
        f"r9={features.rsi_9};"
        f"rsid={features.rsi_speed_divergence};"
        f"m1={features.momentum_1m};"
        f"v30={features.velocity_30s};"
        f"acc={features.momentum_acceleration};"
        f"adx={features.adx_14};"
        f"v5={features.volatility_5m};"
        f"reqv={required_velocity_to_win};"
        f"dt={features.consecutive_directional_ticks};"
        f"ma={_momentum_alignment_text(features)};"
        "t_is_authoritative;"
        f"if_t_gt_240_and_adx_lt_{cfg.discovery_adx_caution_threshold:.0f}_discovery_caution_if_adx_gt_{cfg.trend_priority_adx_threshold:.0f}_prioritize_trend;"
        "DISTANCE_FROM_STRIKE_fields_are_settlement_baseline_positive_means_above_strike_negative_means_below_strike;"
        "MARKET_WIN_CHANCE_fields_are_market_probabilities_not_price_distance;"
        "do_not_confuse_distance_from_strike_with_market_win_chance;"
        "btc_eff_raw_btc_is_true_current_price_for_strike_gap_implied_oracle_is_context_only;"
        "ignore_window_open_drift_v30_is_entry_timing_only;"
        "MARKET_WIN_CHANCE_UP_and_MARKET_WIN_CHANCE_DOWN_are_gamma_market_probabilities;"
        f"if_chosen_market_win_chance_lt_{cfg.market_win_chance_veto_threshold:.2f}_and_15_lte_t_lt_{cfg.market_win_chance_veto_end_seconds}_prefer_no_trade;"
        "if_ma_true_clarity_high_trade_with_trend;"
        f"if_DISTANCE_FROM_STRIKE_USD_beyond_0.5_ATR_in_chosen_direction_may_add_{cfg.itm_confidence_boost_amount:.2f}_confidence;"
        f"if_chosen_side_otm_and_reqv_gt_vol5m_div_{cfg.required_velocity_divisor:.0f}_prefer_no_trade_if_itm_ignore_reqv;"
        "if_t_gt_240_and_abs_DISTANCE_FROM_STRIKE_USD_lt_0.2_ATR_prefer_no_trade_unless_ma_true_and_adx_max;"
        "if_t_gt_60_and_abs_btc_minus_beat_lt_0.2_vol5m_prefer_no_trade;"
        "if_choose_up_and_u_lt_0.45_prefer_no_trade;"
        f"if_r9_gt_{cfg.up_rsi_veto_trend_threshold:.0f}_and_btc_gt_beat_no_up_unless_t_lt_15;"
        f"if_r9_lt_{cfg.down_rsi_veto_threshold:.0f}_no_down_if_r9_gt_{cfg.up_rsi_veto_base_threshold:.0f}_no_up_unless_adx_gt_{cfg.up_rsi_veto_adx_threshold:.0f};"
        "if_rsi_speed_divergence_negative_while_price_up_lower_confidence;"
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
    if not (getattr(cfg, "debug", False) or getattr(cfg, "llm_show_detail", False)):
        return None
    return f"SYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_prompt}"


def _print_probe_prompt(label: str, system_prompt: str, user_prompt: str) -> None:
    print(f"{label} prompt:")
    print("SYSTEM PROMPT:")
    print(system_prompt)
    print()
    print("USER PROMPT:")
    print(user_prompt)


def _request_llm_json_for_probe(
    cfg,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float,
    retry_attempts: int,
    retry_timer_seconds: float,
    probe_label: str,
) -> tuple[dict, str]:
    _print_probe_prompt(probe_label, system_prompt, user_prompt)
    if cfg.engine == "openai":
        raw_text = _request_openai_decision(
            model=cfg.model,
            api_key=cfg.api_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
            retry_timer_seconds=max(retry_timer_seconds, 0.0),
        )
        return _extract_json_payload(raw_text), raw_text

    if cfg.engine == "gemini":
        data, raw_text = _request_gemini_decision_with_parse_retry(
            model=cfg.model,
            api_key=cfg.api_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
            retry_timer_seconds=max(retry_timer_seconds, 0.0),
        )
        return data, raw_text

    raise RuntimeError(f"Unsupported AI engine: {cfg.engine}")


def _validate_probe_response(data: dict, probe_name: str) -> None:
    if str(data.get("status", "")).lower() == "ok":
        return
    decision = str(data.get("decision", "")).upper()
    if decision in {"UP", "DOWN", "NO_TRADE"}:
        return
    raise RuntimeError(f"Unexpected {probe_name} LLM connection test payload: {data}")


def _google_connectivity_detail() -> str:
    is_connected, detail = check_internet_connectivity()
    if is_connected:
        return detail
    return f"FAILED: {detail}"


def _build_basic_connection_test_prompts() -> tuple[str, str]:
    return (
        "You are a connection test for an automated trading agent. "
        "Respond with one JSON object only using this schema: "
        '{"decision":"UP|DOWN|NO_TRADE","confidence":0.0,'
        '"max_price_to_pay":0.0,"reason":"short string"}.',
        'Return exactly {"decision":"NO_TRADE","confidence":0,'
        '"max_price_to_pay":0,"reason":"Connection test"} and nothing else.',
    )


def _build_full_connection_test_prompts() -> tuple[str, str]:
    return (
        _build_openai_realtime_system_prompt(),
        "beat=80818.80297729779;t=0;btc_eff=80823.16;"
        "DISTANCE_FROM_STRIKE_USD=4.359535714989761;"
        "DISTANCE_FROM_STRIKE_PCT=0.0054;"
        "MARKET_WIN_CHANCE_UP=1.0;MARKET_WIN_CHANCE_DOWN=0.01;"
        "up_ask=0.99;down_ask=0.0;"
        "rsi9=37.31201219544613;rsi14=46.585563674512606;"
        "rsi_speed_divergence=-9.273551479066477;"
        "mom1m=-2.3299999900045805;v15=2.940000009999494;"
        "v30=-2.3299999900045805;acc=5.2700000000040745;"
        "adx14=6.828872650974774;atr14=4.403571429286136;"
        "vol5m=7.401005737845983;reqv=None;dir_ticks=3;"
        "ticks10=UUUUDDUUDU;momentum_alignment=False",
    )


def test_llm_connection() -> tuple[bool, str]:
    cfg = get_llm_config()
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

    basic_probe_detail = "basic probe passed"
    try:
        basic_system_prompt, basic_user_prompt = _build_basic_connection_test_prompts()
        basic_data, _ = _request_llm_json_for_probe(
            cfg,
            basic_system_prompt,
            basic_user_prompt,
            api_connection_timeout_seconds,
            api_connection_retry_attempts,
            api_connection_retry_timer_seconds,
            "Basic LLM connection test",
        )
        _validate_probe_response(basic_data, "basic")
    except Exception as exc:
        google_connected, google_detail = check_internet_connectivity()
        basic_probe_detail = f"basic probe failed: {exc}; Google connectivity: {google_detail}"
        if not google_connected:
            return False, basic_probe_detail

    try:
        full_system_prompt, full_user_prompt = _build_full_connection_test_prompts()
        full_data, _ = _request_llm_json_for_probe(
            cfg,
            full_system_prompt,
            full_user_prompt,
            api_connection_timeout_seconds,
            api_connection_retry_attempts,
            api_connection_retry_timer_seconds,
            "Full-size LLM connection test",
        )
        _validate_probe_response(full_data, "full-size")
    except Exception as exc:
        google_detail = _google_connectivity_detail()
        return False, (
            f"{basic_probe_detail}; Full-size LLM probe failed: {exc}; "
            f"Google connectivity: {google_detail}"
        )

    return True, (
        f"LLM connection test succeeded ({cfg.engine}/{cfg.model}; "
        f"{basic_probe_detail}; full-size probe passed)"
    )


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
