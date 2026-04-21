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
    execution_snapshot: Optional["TokenQuoteSnapshot"] = None


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


@dataclass
class AccountBalanceSnapshot:
    signer_address: str
    balance_address: str
    proxy_address: Optional[str]
    cash_balance: Optional[float]
    portfolio_balance: Optional[float]
    total_account_value: Optional[float]
    error: Optional[str]


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


def _normalize_address(address: Optional[str]) -> Optional[str]:
    if not address:
        return None
    if address.startswith("0x"):
        return address.lower()
    return f"0x{address.lower()}"


def _derive_signer_address(private_key: str) -> str:
    from eth_account import Account

    return Account.from_key(private_key).address.lower()


def _balance_of_call_data(address: str) -> str:
    # ERC20 balanceOf(address)
    selector = "70a08231"
    encoded_address = address[2:].rjust(64, "0")
    return f"0x{selector}{encoded_address}"


def _get_polygon_usdc_balance(address: str) -> Optional[float]:
    cfg = get_polymarket_config()
    usdc_address = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"
    rpc_errors = []

    for rpc_url in cfg.polygon_rpc_urls or [cfg.polygon_rpc]:
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [
                {
                    "to": usdc_address,
                    "data": _balance_of_call_data(address),
                },
                "latest",
            ],
            "id": 1,
        }

        try:
            resp = requests.post(rpc_url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            raw_balance = data.get("result")
            if not isinstance(raw_balance, str):
                raise ValueError(f"RPC {rpc_url} returned no result field")
            return int(raw_balance, 16) / 1_000_000
        except Exception as exc:
            rpc_errors.append(f"{rpc_url}: {exc}")

    raise RuntimeError("Failed to fetch Polygon USDC balance from configured RPCs: " + " | ".join(rpc_errors))


def _get_portfolio_value(address: str) -> Optional[float]:
    cfg = get_polymarket_config()
    resp = requests.get(
        f"{cfg.data_api}/value",
        params={"user": address},
        timeout=10,
    )

    if resp.status_code == 404:
        return None

    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return _coerce_price(data[0].get("value"))

    if isinstance(data, dict):
        return _coerce_price(data.get("value"))

    return None


def get_account_balance_snapshot() -> AccountBalanceSnapshot:
    cfg = get_polymarket_config()
    proxy_address = _normalize_address(cfg.proxy_address)
    signer_address = proxy_address or "Unavailable"
    balance_address = proxy_address or "Unavailable"

    cash_balance = None
    portfolio_balance = None
    errors = []

    try:
        if proxy_address:
            try:
                signer_address = _derive_signer_address(cfg.private_key)
            except Exception:
                signer_address = "Unavailable"
        else:
            signer_address = _derive_signer_address(cfg.private_key)
            balance_address = signer_address
    except Exception as exc:
        errors.append(f"Address resolution failed: {exc}")

    if balance_address != "Unavailable":
        try:
            cash_balance = _get_polygon_usdc_balance(balance_address)
        except Exception as exc:
            errors.append(f"Cash balance lookup failed: {exc}")

        try:
            portfolio_balance = _get_portfolio_value(balance_address)
        except Exception as exc:
            errors.append(f"Portfolio balance lookup failed: {exc}")

    total_account_value = None
    if cash_balance is not None and portfolio_balance is not None:
        total_account_value = cash_balance + portfolio_balance

    return AccountBalanceSnapshot(
        signer_address=signer_address,
        balance_address=balance_address,
        proxy_address=proxy_address,
        cash_balance=cash_balance,
        portfolio_balance=portfolio_balance,
        total_account_value=total_account_value,
        error=" | ".join(errors) if errors else None,
    )


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
    snapshot: Optional[TokenQuoteSnapshot] = None,
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
            execution_snapshot=None,
        )

    if decision.confidence < cfg.min_confidence:
        return PaperTradeResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=0.0,
            token_id=None,
            reason=f"Confidence {decision.confidence:.2f} < min {cfg.min_confidence:.2f}",
            execution_snapshot=None,
        )

    token_id = market.up_token_id if decision.side == "UP" else market.down_token_id
    if snapshot is None:
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
            execution_snapshot=snapshot,
        )

    if recommended_limit_price is None:
        return PaperTradeResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=live_price,
            token_id=token_id,
            reason="Could not determine recommended limit price",
            execution_snapshot=snapshot,
        )

    if not snapshot.ok_to_submit:
        return PaperTradeResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=recommended_limit_price,
            token_id=token_id,
            reason=f"Not safe to submit: {snapshot.submit_reason}",
            execution_snapshot=snapshot,
        )

    if live_price > decision.max_price_to_pay:
        return PaperTradeResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=live_price,
            token_id=token_id,
            reason=f"Reference price {live_price:.3f} exceeds LLM max {decision.max_price_to_pay:.3f}",
            execution_snapshot=snapshot,
        )

    if live_price > cfg.max_entry_price:
        return PaperTradeResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=live_price,
            token_id=token_id,
            reason=f"Reference price {live_price:.3f} exceeds max entry {cfg.max_entry_price:.3f}",
            execution_snapshot=snapshot,
        )

    size = cfg.trade_shares_size

    return PaperTradeResult(
        executed=True,
        side=decision.side,
        size=size,
        price=recommended_limit_price,
        token_id=token_id,
        reason=(
            f"Paper trade approved at recommended limit {recommended_limit_price:.3f} "
            f"for {size:.4f} shares "
            f"(reference={live_price:.3f}; {snapshot.submit_reason})"
        ),
        execution_snapshot=snapshot,
    )
