# custom/btc_agent/executor.py

import math
import requests
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Tuple

from agents.polymarket.polymarket import Polymarket
from .config import get_trading_config, get_polymarket_config
from .llm_decision import LlmDecision
from .market_lookup import BtcUpDownMarket
from .network import http_get, http_post


@dataclass
class TradeExecutionResult:
    executed: bool
    side: Optional[str]
    size: float
    price: float
    token_id: Optional[str]
    reason: str
    live_order_response: Optional[Any] = None
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
    legacy_usdc_balance: Optional[float]
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


def _get_polygon_erc20_balance(address: str, token_address: str, token_label: str) -> Optional[float]:
    cfg = get_polymarket_config()
    rpc_errors = []

    for rpc_url in cfg.polygon_rpc_urls or [cfg.polygon_rpc]:
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [
                {
                    "to": token_address,
                    "data": _balance_of_call_data(address),
                },
                "latest",
            ],
            "id": 1,
        }

        try:
            resp = http_post(rpc_url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            raw_balance = data.get("result")
            if not isinstance(raw_balance, str):
                raise ValueError(f"RPC {rpc_url} returned no result field")
            return int(raw_balance, 16) / 1_000_000
        except Exception as exc:
            rpc_errors.append(f"{rpc_url}: {exc}")

    raise RuntimeError(
        f"Failed to fetch Polygon {token_label} balance from configured RPCs: " + " | ".join(rpc_errors)
    )


def _get_polygon_pusd_balance(address: str) -> Optional[float]:
    return _get_polygon_erc20_balance(
        address=address,
        token_address="0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb",
        token_label="pUSD",
    )


def _get_polygon_usdc_balance(address: str) -> Optional[float]:
    return _get_polygon_erc20_balance(
        address=address,
        token_address="0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
        token_label="USDC.e",
    )


def _get_portfolio_value(address: str) -> Optional[float]:
    cfg = get_polymarket_config()
    resp = http_get(
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
    legacy_usdc_balance = None
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
            cash_balance = _get_polygon_pusd_balance(balance_address)
        except Exception as exc:
            errors.append(f"pUSD balance lookup failed: {exc}")

        try:
            legacy_usdc_balance = _get_polygon_usdc_balance(balance_address)
        except Exception as exc:
            errors.append(f"USDC.e balance lookup failed: {exc}")

        try:
            portfolio_balance = _get_portfolio_value(balance_address)
        except Exception as exc:
            errors.append(f"Portfolio balance lookup failed: {exc}")

    total_account_value = None
    if (
        cash_balance is not None
        and legacy_usdc_balance is not None
        and portfolio_balance is not None
    ):
        total_account_value = cash_balance + legacy_usdc_balance + portfolio_balance

    return AccountBalanceSnapshot(
        signer_address=signer_address,
        balance_address=balance_address,
        proxy_address=proxy_address,
        cash_balance=cash_balance,
        legacy_usdc_balance=legacy_usdc_balance,
        portfolio_balance=portfolio_balance,
        total_account_value=total_account_value,
        error=" | ".join(errors) if errors else None,
    )


def _get_price_from_clob_single(token_id: str, side: str) -> Optional[float]:
    cfg = get_polymarket_config()
    url = f"{cfg.clob_api}/price"
    resp = http_get(
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
    resp = http_post(url, json=body, timeout=10)

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
    resp = http_get(url, params={"token_id": token_id}, timeout=10)

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
    resp = http_get(url, params={"token_id": token_id}, timeout=10)

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
    resp = http_get(url, params={"token_id": token_id}, timeout=10)

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


def get_submission_limit_price(snapshot: TokenQuoteSnapshot) -> Optional[float]:
    cfg = get_trading_config()
    if cfg.use_recommended_limit and snapshot.recommended_limit_price is not None:
        return snapshot.recommended_limit_price
    return snapshot.target_limit_price


def get_submission_limit_label() -> str:
    cfg = get_trading_config()
    return "recommended limit" if cfg.use_recommended_limit else "target limit"


def evaluate_ok_to_submit(
    buy_quote: Optional[float],
    submission_limit_price: Optional[float],
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
    limit_label = get_submission_limit_label()

    if submission_limit_price is None:
        return False, f"No {limit_label} available"

    if buy_quote is None:
        return False, "No current buy quote available"

    if buy_quote <= submission_limit_price:
        return True, f"Current buy quote is at or below {limit_label}"

    if tick_size is None or tick_size <= 0:
        return False, (
            f"Current buy quote {buy_quote:.3f} is above {limit_label} "
            f"{submission_limit_price:.3f}"
        )

    allowed_slippage = tick_size * 2
    diff = buy_quote - submission_limit_price

    if diff <= allowed_slippage:
        return True, (
            f"Current buy quote is only {diff:.3f} above {limit_label} "
            f"(within 2 ticks)"
        )

    return False, (
        f"Current buy quote {buy_quote:.3f} moved too far above {limit_label} "
        f"{submission_limit_price:.3f} (diff={diff:.3f}, allowed={allowed_slippage:.3f})"
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
        submission_limit_price=(
            recommended_limit_price
            if get_trading_config().use_recommended_limit
            else target_limit_price
        ),
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


def _build_rejected_trade_result(
    *,
    side: Optional[str],
    size: float,
    price: float,
    token_id: Optional[str],
    reason: str,
    snapshot: Optional[TokenQuoteSnapshot],
) -> TradeExecutionResult:
    return TradeExecutionResult(
        executed=False,
        side=side,
        size=size,
        price=price,
        token_id=token_id,
        reason=reason,
        live_order_response=None,
        execution_snapshot=snapshot,
    )


def _validate_trade_candidate(
    market: BtcUpDownMarket,
    decision: LlmDecision,
    snapshot: Optional[TokenQuoteSnapshot] = None,
) -> Tuple[Optional[TokenQuoteSnapshot], Optional[TradeExecutionResult]]:
    cfg = get_trading_config()

    if decision.side == "NO_TRADE":
        return None, _build_rejected_trade_result(
            side=None,
            size=0.0,
            price=0.0,
            token_id=None,
            reason=f"NO_TRADE from LLM: {decision.reason}",
            snapshot=None,
        )

    if decision.confidence < cfg.min_confidence:
        return None, _build_rejected_trade_result(
            side=decision.side,
            size=0.0,
            price=0.0,
            token_id=None,
            reason=f"Confidence {decision.confidence:.2f} < min {cfg.min_confidence:.2f}",
            snapshot=None,
        )

    token_id = market.up_token_id if decision.side == "UP" else market.down_token_id
    if snapshot is None:
        snapshot = get_token_quote_snapshot(token_id, decision=decision)

    live_price = snapshot.reference_price
    submission_limit_price = get_submission_limit_price(snapshot)
    submission_limit_label = get_submission_limit_label()

    if live_price is None:
        return None, _build_rejected_trade_result(
            side=decision.side,
            size=0.0,
            price=0.0,
            token_id=token_id,
            reason="No buy quote, midpoint, or last trade price available",
            snapshot=snapshot,
        )

    if submission_limit_price is None:
        return None, _build_rejected_trade_result(
            side=decision.side,
            size=0.0,
            price=live_price,
            token_id=token_id,
            reason=f"Could not determine {submission_limit_label}",
            snapshot=snapshot,
        )

    if not snapshot.ok_to_submit:
        return None, _build_rejected_trade_result(
            side=decision.side,
            size=0.0,
            price=submission_limit_price,
            token_id=token_id,
            reason=f"Not safe to submit: {snapshot.submit_reason}",
            snapshot=snapshot,
        )

    if live_price > decision.max_price_to_pay:
        return None, _build_rejected_trade_result(
            side=decision.side,
            size=0.0,
            price=live_price,
            token_id=token_id,
            reason=f"Reference price {live_price:.3f} exceeds LLM max {decision.max_price_to_pay:.3f}",
            snapshot=snapshot,
        )

    if live_price > cfg.max_entry_price:
        return None, _build_rejected_trade_result(
            side=decision.side,
            size=0.0,
            price=live_price,
            token_id=token_id,
            reason=f"Reference price {live_price:.3f} exceeds max entry {cfg.max_entry_price:.3f}",
            snapshot=snapshot,
        )

    return snapshot, None


def _execute_paper_trade(
    decision: LlmDecision,
    snapshot: TokenQuoteSnapshot,
) -> TradeExecutionResult:
    cfg = get_trading_config()
    live_price = snapshot.reference_price
    submission_limit_price = get_submission_limit_price(snapshot)
    submission_limit_label = get_submission_limit_label()
    size = cfg.trade_shares_size
    token_id = snapshot.token_id

    return TradeExecutionResult(
        executed=True,
        side=decision.side,
        size=size,
        price=submission_limit_price,
        token_id=token_id,
        reason=(
            f"Paper trade approved at {submission_limit_label} {submission_limit_price:.3f} "
            f"for {size:.4f} shares "
            f"(reference={live_price:.3f}; {snapshot.submit_reason})"
        ),
        live_order_response=None,
        execution_snapshot=snapshot,
    )


def _estimate_live_fee(size: float, limit_price: float, fee_rate_bps: int) -> float:
    notional = size * limit_price
    return round(notional * (fee_rate_bps / 10_000), 6)


def _get_order_notional(size: float, limit_price: float) -> float:
    return round(size * limit_price, 6)


def _scale_live_size_for_min_notional(
    base_size: float,
    limit_price: float,
    min_order_usd: float,
) -> float:
    if limit_price <= 0:
        raise RuntimeError("Cannot scale live order size with a non-positive limit price.")

    # Add a small buffer above the venue minimum so downstream rounding on the
    # exchange side cannot turn a nominal $1.0000 order into a rejected $0.999x order.
    min_notional_with_buffer = min_order_usd + 0.01
    min_size = min_notional_with_buffer / limit_price
    scaled_size = max(base_size, min_size)
    # Round up to 4 decimals so the post-rounding notional still meets the minimum.
    return math.ceil(scaled_size * 10_000) / 10_000


def _get_required_live_cash(size: float, limit_price: float, fee_rate_bps: int) -> float:
    notional = _get_order_notional(size, limit_price)
    estimated_fee = _estimate_live_fee(size, limit_price, fee_rate_bps)
    return round(notional + estimated_fee, 6)


def ensure_live_trade_cash_available(required_cash: float) -> AccountBalanceSnapshot:
    account = get_account_balance_snapshot()
    if account.cash_balance is None:
        raise RuntimeError(
            "Unable to verify live trading cash balance. Aborting live trading."
        )
    if account.cash_balance < required_cash:
        raise RuntimeError(
            "Not enough cash_balance_pusd to execute live trade: "
            f"required={required_cash:.3f}, available={account.cash_balance:.3f}"
        )
    return account


def _execute_live_trade(
    decision: LlmDecision,
    snapshot: TokenQuoteSnapshot,
) -> TradeExecutionResult:
    cfg = get_trading_config()

    live_price = snapshot.reference_price
    submission_limit_price = get_submission_limit_price(snapshot)
    submission_limit_label = get_submission_limit_label()

    if submission_limit_price is None or live_price is None:
        raise RuntimeError("Live trade execution called without a valid priced snapshot.")

    size = _scale_live_size_for_min_notional(
        cfg.trade_shares_size,
        submission_limit_price,
        cfg.live_min_order_usd,
    )
    order_notional = _get_order_notional(size, submission_limit_price)

    estimated_fee = _estimate_live_fee(size, submission_limit_price, cfg.live_fee_rate_bps)
    required_cash = _get_required_live_cash(
        size,
        submission_limit_price,
        cfg.live_fee_rate_bps,
    )
    ensure_live_trade_cash_available(required_cash)

    client = Polymarket()
    try:
        response = client.execute_order(
            price=submission_limit_price,
            size=size,
            side="BUY",
            token_id=snapshot.token_id,
            fee_rate_bps=cfg.live_fee_rate_bps,
            tick_size=snapshot.tick_size,
        )
    except Exception as exc:
        raise RuntimeError(f"Live order submission failed: {exc}") from exc

    return TradeExecutionResult(
        executed=True,
        side=decision.side,
        size=size,
        price=submission_limit_price,
        token_id=snapshot.token_id,
        reason=(
            f"Live trade submitted at {submission_limit_label} {submission_limit_price:.3f} "
            f"for {size:.4f} shares "
            f"(reference={live_price:.3f}; order_notional={order_notional:.3f}; "
            f"required_cash={required_cash:.3f}; "
            f"estimated_fee={estimated_fee:.3f}; fee_rate_bps={cfg.live_fee_rate_bps}; "
            f"{snapshot.submit_reason})"
        ),
        live_order_response=response,
        execution_snapshot=snapshot,
    )


def maybe_execute_trade(
    market: BtcUpDownMarket,
    decision: LlmDecision,
    snapshot: Optional[TokenQuoteSnapshot] = None,
) -> TradeExecutionResult:
    validated_snapshot, rejection = _validate_trade_candidate(
        market=market,
        decision=decision,
        snapshot=snapshot,
    )
    if rejection is not None:
        return rejection

    assert validated_snapshot is not None

    cfg = get_trading_config()
    if cfg.paper_trading:
        return _execute_paper_trade(decision=decision, snapshot=validated_snapshot)

    return _execute_live_trade(decision=decision, snapshot=validated_snapshot)
