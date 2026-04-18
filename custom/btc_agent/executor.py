# custom/btc_agent/executor.py

import math
import requests
from dataclasses import dataclass
from typing import Optional, Any, Dict, List

from .config import get_trading_config, get_polymarket_config
from .llm_decision import LlmDecision
from .market_lookup import BtcUpDownMarket


@dataclass
class PaperTradeResult:
    executed: bool
    side: Optional[str]
    size: float
    price: float
    token_id: Optional[str]
    reason: str


@dataclass
class TokenQuoteSnapshot:
    token_id: str
    buy_quote: Optional[float]
    midpoint: Optional[float]
    last_trade_price: Optional[float]
    reference_price: Optional[float]
    target_limit_price: Optional[float]
    recommended_limit_price: Optional[float]
    ok_to_submit: bool
    submit_reason: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    tick_size: Optional[float]
    spread: Optional[float]


def _coerce_price(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _extract_single_price(obj: Dict[str, Any]) -> Optional[float]:
    for side_key in ("BUY", "SELL"):
        if side_key in obj:
            price = _coerce_price(obj.get(side_key))
            if price is not None:
                return price

    for field in (
        "price",
        "bestPrice",
        "lastTradePrice",
        "bestAsk",
        "bestBid",
        "mid_price",
        "mid",
    ):
        price = _coerce_price(obj.get(field))
        if price is not None:
            return price

    return None


def _get_price_from_clob_single(token_id: str, side: str) -> Optional[float]:
    cfg = get_polymarket_config()
    url = f"{cfg.clob_api}/price"
    resp = requests.get(
        url,
        params={"token_id": token_id, "side": side},
        timeout=10,
    )

    if resp.status_code == 404:
        return None

    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict):
        return _extract_single_price(data)

    return None


def _get_price_from_clob_multi(token_id: str, side: str) -> Optional[float]:
    cfg = get_polymarket_config()
    url = f"{cfg.clob_api}/prices"
    body: List[Dict[str, Any]] = [{"token_id": token_id, "side": side}]
    resp = requests.post(url, json=body, timeout=10)

    if resp.status_code == 404:
        return None

    resp.raise_for_status()
    data = resp.json()

    key = str(token_id)
    if isinstance(data, dict) and key in data and isinstance(data[key], dict):
        return _extract_single_price(data[key])

    if isinstance(data, dict):
        return _extract_single_price(data)

    return None


def _get_last_trade_price(token_id: str) -> Optional[float]:
    cfg = get_polymarket_config()
    url = f"{cfg.clob_api}/last-trade-price"
    resp = requests.get(url, params={"token_id": token_id}, timeout=10)

    if resp.status_code == 404:
        return None

    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict):
        return _coerce_price(data.get("price"))

    return None


def _get_midpoint_price(token_id: str) -> Optional[float]:
    cfg = get_polymarket_config()
    url = f"{cfg.clob_api}/midpoint"
    resp = requests.get(url, params={"token_id": token_id}, timeout=10)

    if resp.status_code == 404:
        return None

    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict):
        return _coerce_price(data.get("mid_price"))

    return None


def _get_orderbook(token_id: str) -> Dict[str, Any]:
    cfg = get_polymarket_config()
    url = f"{cfg.clob_api}/book"
    resp = requests.get(url, params={"token_id": token_id}, timeout=10)

    if resp.status_code == 404:
        return {}

    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def get_price_for_side(token_id: str, side: str) -> Optional[float]:
    price = _get_price_from_clob_single(token_id, side=side)
    if price is not None:
        return price

    price = _get_price_from_clob_multi(token_id, side=side)
    if price is not None:
        return price

    return None


def compute_reference_price(
    buy_quote: Optional[float],
    midpoint: Optional[float],
    last_trade_price: Optional[float],
    spread: Optional[float],
) -> Optional[float]:
    """
    Suggested decision reference:
    1) midpoint if available and spread isn't extreme
    2) last trade if spread is wide or midpoint missing
    3) buy quote as fallback
    """
    if midpoint is not None:
        if spread is not None and spread > 0.10 and last_trade_price is not None:
            return last_trade_price
        return midpoint

    if last_trade_price is not None:
        return last_trade_price

    if buy_quote is not None:
        return buy_quote

    return None


def _snap_down_to_tick(price: float, tick_size: Optional[float]) -> float:
    """
    Round down to a valid tick increment so the limit is never above target.
    """
    if tick_size is None or tick_size <= 0:
        return round(price, 3)

    ticks = math.floor(price / tick_size)
    snapped = ticks * tick_size
    return round(snapped, 3)


def compute_target_limit_price(
    reference_price: Optional[float],
    decision: Optional[LlmDecision] = None,
) -> Optional[float]:
    if reference_price is None:
        return None

    cfg = get_trading_config()
    target = reference_price

    if decision is not None:
        target = min(target, decision.max_price_to_pay)

    target = min(target, cfg.max_entry_price)
    return round(target, 3)


def compute_recommended_limit_price(
    reference_price: Optional[float],
    tick_size: Optional[float],
    decision: Optional[LlmDecision] = None,
) -> Optional[float]:
    """
    Live-style recommended limit:
    - start from capped target
    - snap down to tick size
    """
    target = compute_target_limit_price(reference_price, decision=decision)
    if target is None:
        return None

    return _snap_down_to_tick(target, tick_size)


def evaluate_ok_to_submit(
    buy_quote: Optional[float],
    recommended_limit_price: Optional[float],
    tick_size: Optional[float],
) -> (bool, str):
    """
    OK to submit if:
    - we have a recommended limit price
    - we have a current buy quote
    - the quote has not moved too far away from the intended limit

    Rule:
    - if buy_quote <= recommended_limit_price, OK
    - else allow up to 2 ticks of adverse movement
    """
    if recommended_limit_price is None:
        return False, "No recommended limit price available"

    if buy_quote is None:
        return False, "No current buy quote available"

    if buy_quote <= recommended_limit_price:
        return True, "Current buy quote is at or below recommended limit"

    if tick_size is None or tick_size <= 0:
        return False, (
            f"Current buy quote {buy_quote:.3f} is above recommended limit "
            f"{recommended_limit_price:.3f}"
        )

    allowed_slippage = tick_size * 2
    diff = buy_quote - recommended_limit_price

    if diff <= allowed_slippage:
        return True, (
            f"Current buy quote is only {diff:.3f} above recommended limit "
            f"(within 2 ticks)"
        )

    return False, (
        f"Current buy quote {buy_quote:.3f} moved too far above recommended limit "
        f"{recommended_limit_price:.3f} (diff={diff:.3f}, allowed={allowed_slippage:.3f})"
    )


def get_token_quote_snapshot(
    token_id: str,
    decision: Optional[LlmDecision] = None,
) -> TokenQuoteSnapshot:
    buy_quote = get_price_for_side(token_id, "BUY")
    midpoint = _get_midpoint_price(token_id)
    last_trade_price = _get_last_trade_price(token_id)

    book = _get_orderbook(token_id)
    bids = book.get("bids") or []
    asks = book.get("asks") or []

    best_bid = None
    best_ask = None

    if bids and isinstance(bids[0], dict):
        best_bid = _coerce_price(bids[0].get("price"))
    if asks and isinstance(asks[0], dict):
        best_ask = _coerce_price(asks[0].get("price"))

    tick_size = _coerce_price(book.get("tick_size"))
    spread = None
    if best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid

    reference_price = compute_reference_price(
        buy_quote=buy_quote,
        midpoint=midpoint,
        last_trade_price=last_trade_price,
        spread=spread,
    )

    target_limit_price = compute_target_limit_price(
        reference_price=reference_price,
        decision=decision,
    )

    recommended_limit_price = compute_recommended_limit_price(
        reference_price=reference_price,
        tick_size=tick_size,
        decision=decision,
    )

    ok_to_submit, submit_reason = evaluate_ok_to_submit(
        buy_quote=buy_quote,
        recommended_limit_price=recommended_limit_price,
        tick_size=tick_size,
    )

    return TokenQuoteSnapshot(
        token_id=token_id,
        buy_quote=buy_quote,
        midpoint=midpoint,
        last_trade_price=last_trade_price,
        reference_price=reference_price,
        target_limit_price=target_limit_price,
        recommended_limit_price=recommended_limit_price,
        ok_to_submit=ok_to_submit,
        submit_reason=submit_reason,
        best_bid=best_bid,
        best_ask=best_ask,
        tick_size=tick_size,
        spread=spread,
    )


def get_best_buy_price(token_id: str) -> Optional[float]:
    snapshot = get_token_quote_snapshot(token_id)
    return snapshot.reference_price


def maybe_execute_paper_trade(
    market: BtcUpDownMarket,
    decision: LlmDecision,
) -> PaperTradeResult:
    cfg = get_trading_config()

    if decision.side == "NO_TRADE":
        return PaperTradeResult(
            executed=False,
            side=None,
            size=0.0,
            price=0.0,
            token_id=None,
            reason=f"NO_TRADE from LLM: {decision.reason}",
        )

    if decision.confidence < cfg.min_confidence:
        return PaperTradeResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=0.0,
            token_id=None,
            reason=f"Confidence {decision.confidence:.2f} < min {cfg.min_confidence:.2f}",
        )

    token_id = market.up_token_id if decision.side == "UP" else market.down_token_id
    snapshot = get_token_quote_snapshot(token_id, decision=decision)

    live_price = snapshot.reference_price
    recommended_limit_price = snapshot.recommended_limit_price

    if live_price is None:
        return PaperTradeResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=0.0,
            token_id=token_id,
            reason="No buy quote, midpoint, or last trade price available",
        )

    if recommended_limit_price is None:
        return PaperTradeResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=live_price,
            token_id=token_id,
            reason="Could not determine recommended limit price",
        )

    if not snapshot.ok_to_submit:
        return PaperTradeResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=recommended_limit_price,
            token_id=token_id,
            reason=f"Not safe to submit: {snapshot.submit_reason}",
        )

    if live_price > decision.max_price_to_pay:
        return PaperTradeResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=live_price,
            token_id=token_id,
            reason=f"Reference price {live_price:.3f} exceeds LLM max {decision.max_price_to_pay:.3f}",
        )

    if live_price > cfg.max_entry_price:
        return PaperTradeResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=live_price,
            token_id=token_id,
            reason=f"Reference price {live_price:.3f} exceeds max entry {cfg.max_entry_price:.3f}",
        )

    size = cfg.max_trade_usd / recommended_limit_price

    return PaperTradeResult(
        executed=True,
        side=decision.side,
        size=size,
        price=recommended_limit_price,
        token_id=token_id,
        reason=(
            f"Paper trade approved at recommended limit {recommended_limit_price:.3f} "
            f"(reference={live_price:.3f}; {snapshot.submit_reason})"
        ),
    )
