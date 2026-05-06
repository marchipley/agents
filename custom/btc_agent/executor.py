# custom/btc_agent/executor.py

import math
import requests
import re
import time
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List, Tuple

from agents.polymarket.polymarket import Polymarket
from .config import get_trading_config, get_polymarket_config
from .indicators import BtcFeatures
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
    quoted_price_at_entry: Optional[float] = None
    actual_fill_price: Optional[float] = None
    realized_slippage_bps: Optional[float] = None
    order_latency_ms: Optional[int] = None
    book_depth_at_fill: Optional[float] = None
    shares_requested: Optional[float] = None


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
    best_bid_size: Optional[float] = None
    best_ask_size: Optional[float] = None
    spread_bps: Optional[float] = None
    top_level_book_imbalance: Optional[float] = None
    imbalance_pressure: Optional[float] = None


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


def _fmt_price_debug(value: Optional[float]) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "None"


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


def _extract_level_size(level: Dict[str, Any]) -> Optional[float]:
    for field in ("size", "quantity", "amount", "asset_size", "shares"):
        size = _coerce_price(level.get(field))
        if size is not None:
            return size
    if "level" in level and isinstance(level["level"], dict):
        return _extract_level_size(level["level"])
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

    return round(reference_price, 3)


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
    if getattr(cfg, "use_recommended_limit", True) and snapshot.recommended_limit_price is not None:
        return snapshot.recommended_limit_price
    return snapshot.target_limit_price


def get_submission_limit_label() -> str:
    cfg = get_trading_config()
    return "recommended limit" if cfg.use_recommended_limit else "target limit"


def evaluate_ok_to_submit(
    buy_quote: Optional[float],
    reference_price: Optional[float],
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

    cfg = get_trading_config()
    if not cfg.use_recommended_limit:
        if reference_price is None:
            return False, "No reference price available"
        if reference_price <= submission_limit_price:
            return True, f"Reference price is at or below {limit_label}"
        return False, (
            f"Reference price {reference_price:.3f} exceeds {limit_label} "
            f"{submission_limit_price:.3f}"
        )

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
    best_bid_size = None
    best_ask_size = None
    if bids and isinstance(bids[0], dict):
        best_bid_size = _extract_level_size(bids[0])
    if asks and isinstance(asks[0], dict):
        best_ask_size = _extract_level_size(asks[0])

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
    spread_bps = None
    if spread is not None and reference_price not in (None, 0):
        spread_bps = (spread / reference_price) * 10_000
    top_level_book_imbalance = None
    top_depth_levels = 5
    top_three_bid_size = sum(
        size
        for size in (_extract_level_size(level) for level in bids[:top_depth_levels] if isinstance(level, dict))
        if size is not None
    )
    top_three_ask_size = sum(
        size
        for size in (_extract_level_size(level) for level in asks[:top_depth_levels] if isinstance(level, dict))
        if size is not None
    )
    imbalance_pressure = None
    total_top_three_size = top_three_bid_size + top_three_ask_size
    if total_top_three_size > 0:
        top_level_book_imbalance = top_three_bid_size / total_top_three_size
        imbalance_pressure = (top_three_bid_size - top_three_ask_size) / total_top_three_size

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
        reference_price=reference_price,
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
        best_bid_size=best_bid_size,
        best_ask_size=best_ask_size,
        spread_bps=spread_bps,
        top_level_book_imbalance=top_level_book_imbalance,
        imbalance_pressure=imbalance_pressure,
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
    quoted_price_at_entry: Optional[float] = None,
    actual_fill_price: Optional[float] = None,
    realized_slippage_bps: Optional[float] = None,
    order_latency_ms: Optional[int] = None,
    book_depth_at_fill: Optional[float] = None,
    shares_requested: Optional[float] = None,
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
        quoted_price_at_entry=quoted_price_at_entry,
        actual_fill_price=actual_fill_price,
        realized_slippage_bps=realized_slippage_bps,
        order_latency_ms=order_latency_ms,
        book_depth_at_fill=book_depth_at_fill,
        shares_requested=shares_requested,
    )


def _get_book_depth_at_fill(snapshot: Optional[TokenQuoteSnapshot]) -> Optional[float]:
    if snapshot is None:
        return None
    for value in (
        getattr(snapshot, "best_ask_size", None),
        getattr(snapshot, "best_bid_size", None),
    ):
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _compute_realized_slippage_bps(
    quoted_price_at_entry: Optional[float],
    actual_fill_price: Optional[float],
) -> Optional[float]:
    if quoted_price_at_entry in (None, 0) or actual_fill_price is None:
        return None
    try:
        quoted_price = float(quoted_price_at_entry)
        fill_price = float(actual_fill_price)
    except (TypeError, ValueError):
        return None
    if quoted_price == 0:
        return None
    return ((fill_price - quoted_price) / quoted_price) * 10_000


def _extract_order_id_from_live_response(response: Any) -> Optional[str]:
    if response is None:
        return None
    if isinstance(response, dict):
        for key in ("orderID", "orderId", "id"):
            value = response.get(key)
            if value:
                return str(value)
        for key in ("data", "order", "result"):
            nested = response.get(key)
            order_id = _extract_order_id_from_live_response(nested)
            if order_id:
                return order_id
        return None
    if isinstance(response, (list, tuple)):
        for item in response:
            order_id = _extract_order_id_from_live_response(item)
            if order_id:
                return order_id
        return None
    return None


def _extract_average_fill_price_from_live_response(response: Any) -> Optional[float]:
    if response is None:
        return None
    if isinstance(response, dict):
        for key in (
            "avgPrice",
            "averagePrice",
            "average_price",
            "avg_fill_price",
            "avg_fill",
            "fillPrice",
            "filledPrice",
            "price",
        ):
            value = _coerce_price(response.get(key))
            if value is not None:
                return value
        for key in ("data", "order", "result"):
            nested = response.get(key)
            average_price = _extract_average_fill_price_from_live_response(nested)
            if average_price is not None:
                return average_price
        return None
    if isinstance(response, (list, tuple)):
        prices = [
            _extract_average_fill_price_from_live_response(item)
            for item in response
        ]
        prices = [price for price in prices if price is not None]
        return prices[0] if prices else None
    return None


def _weighted_average_fill_price(trades: List[Dict[str, Any]]) -> Optional[float]:
    total_notional = 0.0
    total_size = 0.0
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        price = None
        size = None
        for key in ("price", "tradePrice", "fillPrice"):
            price = _coerce_price(trade.get(key))
            if price is not None:
                break
        for key in ("size", "amount", "asset_size", "shares", "filledSize"):
            size = _coerce_price(trade.get(key))
            if size is not None:
                break
        if price is None or size is None or size <= 0:
            continue
        total_notional += price * size
        total_size += size
    if total_size <= 0:
        return None
    return total_notional / total_size


def _fetch_actual_fill_price_from_trades(
    order_id: str,
    token_id: Optional[str],
) -> Optional[float]:
    cfg = get_polymarket_config()
    candidate_params = [
        {"orderID": order_id},
        {"orderId": order_id},
        {"order_id": order_id},
    ]
    if token_id:
        candidate_params.extend(
            [
                {"maker_order_id": order_id, "asset_id": token_id},
                {"taker_order_id": order_id, "asset_id": token_id},
            ]
        )

    for params in candidate_params:
        try:
            resp = http_get(f"{cfg.data_api}/trades", params=params, timeout=10)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        trades: List[Dict[str, Any]]
        if isinstance(data, list):
            trades = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            if isinstance(data.get("data"), list):
                trades = [item for item in data["data"] if isinstance(item, dict)]
            elif isinstance(data.get("trades"), list):
                trades = [item for item in data["trades"] if isinstance(item, dict)]
            else:
                trades = [data]
        else:
            trades = []

        average_fill = _weighted_average_fill_price(trades)
        if average_fill is not None:
            return average_fill
    return None


def _resolve_actual_fill_price(response: Any, token_id: Optional[str]) -> Optional[float]:
    average_fill = _extract_average_fill_price_from_live_response(response)
    if average_fill is not None:
        return average_fill
    order_id = _extract_order_id_from_live_response(response)
    if not order_id:
        return None
    return _fetch_actual_fill_price_from_trades(order_id, token_id)


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


def _get_time_remaining_seconds(market: BtcUpDownMarket) -> int:
    now_ts = int(datetime.now(timezone.utc).timestamp())
    slug_start_ts = _slug_start_ts(getattr(market, "slug", None))
    canonical_start_ts = slug_start_ts or getattr(market, "start_ts", None)
    canonical_end_ts = canonical_start_ts + 300 if canonical_start_ts else None
    effective_end_ts = getattr(market, "end_ts", None)

    # BTC 5-minute markets are timestamp-aligned. Some upstream payloads can surface
    # a stale or already-expired end_ts during slug rollover, so prefer the canonical
    # 5-minute boundary derived from the slug-aligned start when it is later.
    if canonical_end_ts is not None:
        if effective_end_ts is None or canonical_end_ts > effective_end_ts:
            effective_end_ts = canonical_end_ts

    if effective_end_ts is None:
        return 0

    return max(int(effective_end_ts) - now_ts, 0)


def _get_implied_probability(snapshot: TokenQuoteSnapshot) -> Optional[float]:
    for value in (snapshot.buy_quote, snapshot.best_ask, snapshot.reference_price):
        if value is None:
            continue
        if 0 <= value <= 1:
            return value
    return None


def _get_effective_fee_probability(implied_probability: float) -> float:
    # Polymarket fee impact on high-probability favorites is much smaller than a raw
    # p * (1-p) penalty suggests. Use a scaled version so fees decay near the extremes
    # instead of overwhelming high-confidence late-window signals.
    return implied_probability * (1 - implied_probability) * 0.1


def _compute_execution_edge(decision: LlmDecision, snapshot: TokenQuoteSnapshot) -> Optional[float]:
    return _compute_execution_edge_for_confidence(decision.confidence, snapshot)


def _compute_execution_edge_for_confidence(confidence: float, snapshot: TokenQuoteSnapshot) -> Optional[float]:
    implied_probability = _get_implied_probability(snapshot)
    if implied_probability is None:
        return None
    return confidence - (implied_probability + _get_effective_fee_probability(implied_probability))


def _get_market_implied_probability(
    market: BtcUpDownMarket,
    decision: LlmDecision,
    snapshot: TokenQuoteSnapshot,
) -> Optional[float]:
    if decision.side == "UP":
        probability = getattr(market, "up_market_probability", None)
    elif decision.side == "DOWN":
        probability = getattr(market, "down_market_probability", None)
    else:
        probability = None
    if probability is not None:
        return float(probability)
    if snapshot.buy_quote is not None:
        return float(snapshot.buy_quote)
    return _get_implied_probability(snapshot)


def _get_strike_delta_pct(features: Optional[BtcFeatures], market: BtcUpDownMarket) -> Optional[float]:
    if (
        features is None
        or getattr(features, "price_usd", None) in (None, 0)
        or market.settlement_threshold in (None, 0)
    ):
        return None
    gap_to_target = float(features.price_usd) - float(market.settlement_threshold)
    return gap_to_target / float(features.price_usd)


def get_effective_decision_confidence(
    decision: LlmDecision,
    market: BtcUpDownMarket,
    features: Optional[BtcFeatures] = None,
) -> float:
    confidence = float(getattr(decision, "confidence", 0.0) or 0.0)
    strike_delta_pct = _get_strike_delta_pct(features, market)
    up_probability = getattr(market, "up_market_probability", None)
    down_probability = getattr(market, "down_market_probability", None)

    if (
        decision.side == "UP"
        and strike_delta_pct is not None
        and strike_delta_pct > 0.0002
        and up_probability is not None
        and float(up_probability) > 0.60
    ):
        confidence = min(confidence + 0.10, 1.0)
    elif (
        decision.side == "DOWN"
        and strike_delta_pct is not None
        and strike_delta_pct < -0.0002
        and down_probability is not None
        and float(down_probability) > 0.60
    ):
        confidence = min(confidence + 0.10, 1.0)

    return confidence


def get_effective_min_confidence(
    market: BtcUpDownMarket,
    features: Optional[BtcFeatures] = None,
    cfg=None,
) -> float:
    cfg = cfg or get_trading_config()
    base_confidence = float(getattr(cfg, "min_confidence", 0.7))
    time_remaining_seconds = _get_time_remaining_seconds(market)
    adx_14 = None if features is None else getattr(features, "adx_14", None)

    if time_remaining_seconds < 60:
        return max(base_confidence, 0.75)
    if adx_14 is not None and adx_14 > 35:
        return min(base_confidence, 0.62)
    return base_confidence


def _is_high_price_trade(snapshot: TokenQuoteSnapshot) -> bool:
    implied_probability = _get_implied_probability(snapshot)
    if implied_probability is None:
        return False
    return implied_probability > 0.80


def _is_window_delta_master_switch(
    features: Optional[BtcFeatures],
    time_remaining_seconds: int,
) -> bool:
    if features is None or time_remaining_seconds > 10:
        return False
    return abs(features.delta_pct_from_window_open) > 0.0015


def _validate_trade_candidate(
    market: BtcUpDownMarket,
    decision: LlmDecision,
    features: Optional[BtcFeatures] = None,
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

    token_id = market.up_token_id if decision.side == "UP" else market.down_token_id
    if snapshot is None:
        snapshot = get_token_quote_snapshot(token_id, decision=decision)

    live_price = snapshot.reference_price
    submission_limit_price = get_submission_limit_price(snapshot)
    submission_limit_label = get_submission_limit_label()
    effective_confidence: Optional[float] = None

    def _reject(price: float, reason: str) -> Tuple[None, TradeExecutionResult]:
        return None, _build_rejected_trade_result(
            side=decision.side,
            size=0.0,
            price=price,
            token_id=token_id,
            reason=reason,
            snapshot=snapshot,
            **_get_rejection_intent_context(
                decision,
                snapshot,
                submission_limit_price,
                effective_confidence=effective_confidence,
            ),
        )

    if live_price is None:
        return _reject(0.0, "No buy quote, midpoint, or last trade price available")

    if submission_limit_price is None:
        return _reject(live_price, f"Could not determine {submission_limit_label}")

    if not snapshot.ok_to_submit:
        return _reject(submission_limit_price, f"Not safe to submit: {snapshot.submit_reason}")

    implied_probability = _get_implied_probability(snapshot)
    if implied_probability is None:
        return _reject(live_price, "Could not determine implied probability from market quote")

    effective_confidence = get_effective_decision_confidence(
        decision,
        market,
        features=features,
    )
    min_confidence = get_effective_min_confidence(
        market,
        features=features,
        cfg=cfg,
    )

    if effective_confidence < min_confidence:
        return _reject(
            live_price,
            (
                "Confidence floor veto blocked directional trade "
                f"(confidence={decision.confidence:.3f}; "
                f"effective_confidence={effective_confidence:.3f}; "
                f"min_confidence={min_confidence:.3f})"
            ),
        )

    if effective_confidence <= 0:
        return _reject(live_price, "Confidence must be positive for directional execution")

    edge = _compute_execution_edge_for_confidence(effective_confidence, snapshot)
    if edge is None:
        return _reject(live_price, "Could not compute execution edge")

    time_remaining_seconds = _get_time_remaining_seconds(market)
    hard_deadline_execution = time_remaining_seconds < 5 and effective_confidence > 0.70
    high_confidence_override = effective_confidence > 0.90
    window_delta_master_switch = _is_window_delta_master_switch(features, time_remaining_seconds)
    min_edge_required = 0.0 if high_confidence_override else 0.05
    chosen_side_quote = snapshot.buy_quote if snapshot.buy_quote is not None else implied_probability
    gap_to_target = None
    if features is not None and getattr(features, "price_usd", None) is not None and market.settlement_threshold not in (None, 0):
        gap_to_target = float(features.price_usd) - float(market.settlement_threshold)
    required_velocity_to_win = None
    if gap_to_target is not None and time_remaining_seconds > 0:
        required_velocity_to_win = abs(gap_to_target) / time_remaining_seconds
    volatility_5m = None if features is None else getattr(features, "volatility_5m", None)
    rsi_9 = None if features is None else getattr(features, "rsi_9", None)
    market_implied_probability = _get_market_implied_probability(market, decision, snapshot)
    chosen_side_market_win_chance = (
        float(market_implied_probability)
        if market_implied_probability is not None
        else chosen_side_quote
    )
    consensus_gap = (
        None
        if market_implied_probability is None
        else abs(float(effective_confidence) - float(market_implied_probability))
    )

    if (
        decision.side == "DOWN"
        and rsi_9 is not None
        and rsi_9 < 30
    ):
        return _reject(
            submission_limit_price,
            (
                "RSI directional veto blocked DOWN trade in oversold conditions "
                f"(rsi_9={rsi_9:.3f})"
            ),
        )

    if (
        decision.side == "UP"
        and rsi_9 is not None
        and rsi_9 > 70
    ):
        return _reject(
            submission_limit_price,
            (
                "RSI directional veto blocked UP trade in overbought conditions "
                f"(rsi_9={rsi_9:.3f})"
            ),
        )

    if (
        chosen_side_market_win_chance is not None
        and chosen_side_market_win_chance < 0.10
        and time_remaining_seconds > 180
    ):
        return _reject(
            submission_limit_price,
            (
                "Discovery-phase quote-floor veto blocked extremely low-probability trade "
                f"(market_win_chance={chosen_side_market_win_chance:.3f}; time_remaining={time_remaining_seconds}s)"
            ),
        )

    if (
        chosen_side_market_win_chance is not None
        and chosen_side_market_win_chance < 0.15
        and 15 <= time_remaining_seconds < 120
    ):
        return _reject(
            submission_limit_price,
            (
                "Quote-floor veto blocked low-probability reversal trade "
                f"(market_win_chance={chosen_side_market_win_chance:.3f}; time_remaining={time_remaining_seconds}s)"
            ),
        )

    if (
        required_velocity_to_win is not None
        and volatility_5m not in (None, 0)
        and required_velocity_to_win > (float(volatility_5m) / 10.0)
    ):
        return _reject(
            submission_limit_price,
            (
                "Velocity/volatility veto blocked trade "
                f"(required_velocity_to_win={required_velocity_to_win:.3f}; "
                f"volatility_5m={float(volatility_5m):.3f}; "
                f"threshold={(float(volatility_5m) / 10.0):.3f})"
            ),
        )

    if (
        consensus_gap is not None
        and consensus_gap > 0.50
    ):
        return _reject(
            submission_limit_price,
            (
                "Consensus-gap veto blocked hallucinated edge "
                f"(confidence={decision.confidence:.3f}; "
                f"effective_confidence={effective_confidence:.3f}; "
                f"market_probability={market_implied_probability:.3f}; "
                f"consensus_gap={consensus_gap:.3f})"
            ),
        )

    if (
        gap_to_target is not None
        and volatility_5m not in (None, 0)
        and time_remaining_seconds > 60
        and abs(gap_to_target) < (float(volatility_5m) * 0.2)
    ):
        return _reject(
            submission_limit_price,
            (
                "Too close to call: target gap is inside the victory-margin buffer "
                f"(gap={gap_to_target:.3f}; volatility_5m={float(volatility_5m):.3f}; "
                f"buffer={(float(volatility_5m) * 0.2):.3f}; time_remaining={time_remaining_seconds}s)"
            ),
        )

    if (
        decision.side == "UP"
        and chosen_side_quote is not None
        and chosen_side_quote < 0.45
    ):
        return _reject(
            submission_limit_price,
            (
                "Quote-price divergence veto blocked UP trade "
                f"(up_buy_quote={chosen_side_quote:.3f})"
            ),
        )

    if (
        decision.side == "UP"
        and rsi_9 is not None
        and rsi_9 > 85
        and gap_to_target is not None
        and gap_to_target > 0
    ):
        return _reject(
            submission_limit_price,
            (
                "RSI ceiling veto blocked UP trade above strike "
                f"(rsi_9={rsi_9:.3f}; gap={gap_to_target:.3f})"
            ),
        )

    if (
        not getattr(cfg, "disable_liquidity_filter", False)
        and snapshot.spread_bps is not None
        and snapshot.spread_bps > 150
    ):
        return _reject(
            live_price,
            (
                "Thin liquidity blocked execution "
                f"(spread_bps={snapshot.spread_bps:.1f})"
            ),
        )

    if (
        not window_delta_master_switch
        and
        not getattr(cfg, "disable_liquidity_filter", False)
        and _is_high_price_trade(snapshot)
        and (market.volume is None or market.volume <= 1000)
    ):
        return _reject(
            live_price,
            (
                "High-price trade blocked by liquidity filter "
                f"(implied_probability={implied_probability:.3f}; volume={market.volume})"
            ),
        )

    if not hard_deadline_execution and not window_delta_master_switch and edge <= min_edge_required:
        return _reject(
            live_price,
            (
                f"Execution edge {edge:.3f} <= {min_edge_required:.3f} "
                f"(confidence={decision.confidence:.3f}; "
                f"effective_confidence={effective_confidence:.3f}; "
                f"implied_probability={implied_probability:.3f})"
            ),
        )

    return snapshot, None


def _execute_paper_trade(
    decision: LlmDecision,
    snapshot: TokenQuoteSnapshot,
    effective_confidence: Optional[float] = None,
) -> TradeExecutionResult:
    cfg = get_trading_config()
    live_price = snapshot.reference_price
    submission_limit_price = get_submission_limit_price(snapshot)
    submission_limit_label = get_submission_limit_label()
    max_order_budget_usd = _get_max_order_budget_usd(cfg)
    token_id = snapshot.token_id
    if submission_limit_price is None or live_price is None:
        raise RuntimeError("Paper trade execution called without a valid priced snapshot.")
    decision_confidence = float(
        decision.confidence if effective_confidence is None else effective_confidence
    )
    size, used_high_confidence_override = _get_order_size_for_decision(
        decision_confidence,
        cfg,
        submission_limit_price,
    )
    quoted_price_at_entry = snapshot.buy_quote
    actual_fill_price = submission_limit_price
    realized_slippage_bps = _compute_realized_slippage_bps(
        quoted_price_at_entry,
        actual_fill_price,
    )
    book_depth_at_fill = _get_book_depth_at_fill(snapshot)
    order_notional = _get_order_notional(size, submission_limit_price)
    if size <= 0 or order_notional <= 0:
        return TradeExecutionResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=submission_limit_price,
            token_id=token_id,
            reason=(
                f"Order budget {max_order_budget_usd:.3f} is too small for "
                f"{submission_limit_label} {submission_limit_price:.3f}"
            ),
            live_order_response=None,
            execution_snapshot=snapshot,
            quoted_price_at_entry=quoted_price_at_entry,
            actual_fill_price=actual_fill_price,
            realized_slippage_bps=realized_slippage_bps,
            order_latency_ms=0,
            book_depth_at_fill=book_depth_at_fill,
            shares_requested=size,
        )

    return TradeExecutionResult(
        executed=True,
        side=decision.side,
        size=size,
        price=submission_limit_price,
        token_id=token_id,
        reason=(
            f"Paper trade approved at {submission_limit_label} {submission_limit_price:.3f} "
            f"for {size:.4f} shares "
            f"(reference={live_price:.3f}; order_notional={order_notional:.3f}; "
            f"quoted_price_at_entry={_fmt_price_debug(quoted_price_at_entry)}; "
            f"actual_fill_price={_fmt_price_debug(actual_fill_price)}; "
            f"realized_slippage_bps={_fmt_price_debug(realized_slippage_bps)}; "
            f"book_depth_at_fill={_fmt_price_debug(book_depth_at_fill)}; "
            f"max_order_price_usd={max_order_budget_usd:.3f}; "
            f"high_confidence_size_override={used_high_confidence_override}; "
            f"{snapshot.submit_reason})"
        ),
        live_order_response=None,
        execution_snapshot=snapshot,
        quoted_price_at_entry=quoted_price_at_entry,
        actual_fill_price=actual_fill_price,
        realized_slippage_bps=realized_slippage_bps,
        order_latency_ms=0,
        book_depth_at_fill=book_depth_at_fill,
        shares_requested=size,
    )


def _estimate_live_fee(size: float, limit_price: float, fee_rate_bps: int) -> float:
    notional = size * limit_price
    return round(notional * (fee_rate_bps / 10_000), 6)


def _get_order_notional(size: float, limit_price: float) -> float:
    return round(size * limit_price, 6)


def _size_for_max_budget(max_budget_usd: float, limit_price: float) -> float:
    if limit_price <= 0:
        raise RuntimeError("Cannot size an order with a non-positive limit price.")
    if max_budget_usd <= 0:
        return 0.0
    raw_size = max_budget_usd / limit_price
    return math.floor(raw_size * 10_000) / 10_000


def _quantize_live_buy_size_for_amount_precision(price: float, size: float) -> float:
    """
    Polymarket BUY orders require quote-side (maker) amount precision of 2 decimals
    and token-side (taker) amount precision of 4 decimals.

    To satisfy both constraints, round size down to the nearest 4-decimal quantum
    that makes price * size representable to cents.
    """
    if price <= 0 or size <= 0:
        return 0.0

    price_dec = Decimal(str(price)).normalize()
    size_dec = Decimal(str(size)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
    exponent = price_dec.as_tuple().exponent
    price_decimals = -exponent if exponent < 0 else 0
    scale = 10 ** price_decimals
    price_int = int((price_dec * scale).to_integral_value())

    required_divisor = 10 ** (price_decimals + 2)
    gcd_value = math.gcd(price_int, required_divisor)
    quantum_numerator = required_divisor // gcd_value
    quantum = Decimal(quantum_numerator) / Decimal(10_000)
    if quantum <= 0:
        return float(size_dec)

    units = (size_dec / quantum).to_integral_value(rounding=ROUND_DOWN)
    quantized_size = units * quantum
    return float(quantized_size.quantize(Decimal("0.0001"), rounding=ROUND_DOWN))


def _get_max_order_budget_usd(cfg) -> float:
    if hasattr(cfg, "max_order_price_usd"):
        return max(float(cfg.max_order_price_usd), 0.0)
    if hasattr(cfg, "trade_shares_size"):
        return max(float(cfg.trade_shares_size), 0.0)
    return 0.0


def _get_rejection_intent_context(
    decision: Optional[LlmDecision],
    snapshot: Optional[TokenQuoteSnapshot],
    submission_limit_price: Optional[float],
    effective_confidence: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    quoted_price_at_entry = None if snapshot is None else snapshot.buy_quote
    book_depth_at_fill = _get_book_depth_at_fill(snapshot)
    shares_requested = None
    if decision is not None and submission_limit_price not in (None, 0):
        cfg = get_trading_config()
        decision_confidence = float(
            decision.confidence if effective_confidence is None else effective_confidence
        )
        shares_requested, _ = _get_order_size_for_decision(
            decision_confidence,
            cfg,
            float(submission_limit_price),
        )
        if not getattr(cfg, "paper_trading", True):
            shares_requested = _quantize_live_buy_size_for_amount_precision(
                float(submission_limit_price),
                float(shares_requested),
            )
    return {
        "quoted_price_at_entry": quoted_price_at_entry,
        "actual_fill_price": None,
        "realized_slippage_bps": None,
        "order_latency_ms": 0,
        "book_depth_at_fill": book_depth_at_fill,
        "shares_requested": shares_requested,
    }


def _get_order_size_for_decision(
    decision_confidence: float,
    cfg,
    submission_limit_price: float,
) -> tuple[float, bool]:
    threshold = getattr(cfg, "max_size_high_confidence_threshold", 1.1)
    override_shares = max(getattr(cfg, "max_size_high_confidence_shares", 0.0), 0.0)
    if override_shares > 0 and decision_confidence >= threshold:
        return override_shares, True
    return _size_for_max_budget(_get_max_order_budget_usd(cfg), submission_limit_price), False


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


def _is_fok_full_fill_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "couldn't be fully filled" in message or "fully filled or killed" in message


def _extract_minimum_size_from_error(exc: Exception) -> Optional[float]:
    match = re.search(r"minimum:\s*([0-9]+(?:\.[0-9]+)?)", str(exc), re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


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
    market: BtcUpDownMarket,
    snapshot: TokenQuoteSnapshot,
    effective_confidence: Optional[float] = None,
) -> TradeExecutionResult:
    cfg = get_trading_config()

    live_price = snapshot.reference_price
    submission_limit_price = get_submission_limit_price(snapshot)
    submission_limit_label = get_submission_limit_label()
    max_order_budget_usd = _get_max_order_budget_usd(cfg)
    time_remaining_seconds = _get_time_remaining_seconds(market)
    use_fok = time_remaining_seconds <= 10
    order_type_label = "FOK" if use_fok else "GTC"
    implied_probability = _get_implied_probability(snapshot)
    decision_confidence = float(
        decision.confidence if effective_confidence is None else effective_confidence
    )
    edge = _compute_execution_edge_for_confidence(decision_confidence, snapshot)

    if submission_limit_price is None or live_price is None:
        raise RuntimeError("Live trade execution called without a valid priced snapshot.")

    size, used_high_confidence_override = _get_order_size_for_decision(
        decision_confidence,
        cfg,
        submission_limit_price,
    )
    size = _quantize_live_buy_size_for_amount_precision(submission_limit_price, size)
    quoted_price_at_entry = snapshot.buy_quote
    book_depth_at_fill = _get_book_depth_at_fill(snapshot)
    min_order_size = _scale_live_size_for_min_notional(
        0.0,
        submission_limit_price,
        cfg.live_min_order_usd,
    )
    if size <= 0:
        return TradeExecutionResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=submission_limit_price,
            token_id=snapshot.token_id,
            reason=(
                f"Order budget {max_order_budget_usd:.3f} is too small for "
                f"{submission_limit_label} {submission_limit_price:.3f}"
            ),
            live_order_response=None,
            execution_snapshot=snapshot,
            quoted_price_at_entry=quoted_price_at_entry,
            order_latency_ms=0,
            book_depth_at_fill=book_depth_at_fill,
            shares_requested=size,
        )
    if size < min_order_size and not used_high_confidence_override:
        return TradeExecutionResult(
            executed=False,
            side=decision.side,
            size=0.0,
            price=submission_limit_price,
            token_id=snapshot.token_id,
            reason=(
                "Order budget cannot satisfy live minimum order size "
                f"(max_order_price_usd={max_order_budget_usd:.3f}; "
                f"{submission_limit_label}={submission_limit_price:.3f}; "
                f"max_size={size:.4f}; required_min_size={min_order_size:.4f})"
            ),
            live_order_response=None,
            execution_snapshot=snapshot,
            quoted_price_at_entry=quoted_price_at_entry,
            order_latency_ms=0,
            book_depth_at_fill=book_depth_at_fill,
            shares_requested=size,
        )
    client = Polymarket()

    def _submit_order(order_size: float, fok_enabled: bool):
        order_notional_local = _get_order_notional(order_size, submission_limit_price)
        estimated_fee_local = _estimate_live_fee(order_size, submission_limit_price, cfg.live_fee_rate_bps)
        required_cash_local = _get_required_live_cash(
            order_size,
            submission_limit_price,
            cfg.live_fee_rate_bps,
        )
        ensure_live_trade_cash_available(required_cash_local)
        submit_started = time.monotonic()
        response_local = client.execute_order(
            price=submission_limit_price,
            size=order_size,
            side="BUY",
            token_id=snapshot.token_id,
            fee_rate_bps=cfg.live_fee_rate_bps,
            tick_size=snapshot.tick_size,
            use_fok=fok_enabled,
        )
        latency_ms_local = int(round((time.monotonic() - submit_started) * 1000))
        return response_local, order_notional_local, estimated_fee_local, required_cash_local, latency_ms_local

    order_latency_ms = 0
    try:
        response, order_notional, estimated_fee, required_cash, order_latency_ms = _submit_order(size, use_fok)
    except Exception as exc:
        minimum_size = _extract_minimum_size_from_error(exc)
        if minimum_size is not None and minimum_size > size:
            return TradeExecutionResult(
                executed=False,
                side=decision.side,
                size=0.0,
                price=submission_limit_price,
                token_id=snapshot.token_id,
                reason=(
                    "Exchange minimum size exceeds configured order budget "
                    f"(max_order_price_usd={max_order_budget_usd:.3f}; "
                    f"attempted_size={size:.4f}; exchange_minimum_size={minimum_size:.4f}; "
                    f"{submission_limit_label}={submission_limit_price:.3f})"
                ),
                live_order_response=None,
                execution_snapshot=snapshot,
                quoted_price_at_entry=quoted_price_at_entry,
                order_latency_ms=0,
                book_depth_at_fill=book_depth_at_fill,
                shares_requested=size,
            )
        elif use_fok and _is_fok_full_fill_error(exc):
            if time_remaining_seconds > 5:
                try:
                    response, order_notional, estimated_fee, required_cash, order_latency_ms = _submit_order(size, False)
                    order_type_label = "GTC (after FOK retry)"
                except Exception as retry_exc:
                    raise RuntimeError(
                        "Live order submission failed after FOK retry: "
                        f"initial={exc}; retry={retry_exc}"
                    ) from retry_exc
            else:
                return TradeExecutionResult(
                    executed=False,
                    side=decision.side,
                    size=0.0,
                    price=submission_limit_price,
                    token_id=snapshot.token_id,
                    reason=(
                        "FOK order could not be fully filled in the final deadline window "
                        f"(time_remaining={time_remaining_seconds}s; {snapshot.submit_reason})"
                    ),
                    live_order_response=None,
                    execution_snapshot=snapshot,
                    quoted_price_at_entry=quoted_price_at_entry,
                    order_latency_ms=0,
                    book_depth_at_fill=book_depth_at_fill,
                    shares_requested=size,
                )
        else:
            raise RuntimeError(f"Live order submission failed: {exc}") from exc

    actual_fill_price = _resolve_actual_fill_price(response, snapshot.token_id)
    realized_slippage_bps = _compute_realized_slippage_bps(
        quoted_price_at_entry,
        actual_fill_price,
    )

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
            f"implied_probability={implied_probability:.3f}; "
            f"edge={edge:.3f}; "
            f"time_remaining={time_remaining_seconds}s; "
            f"order_type={order_type_label}; "
            f"quoted_price_at_entry={_fmt_price_debug(quoted_price_at_entry)}; "
            f"actual_fill_price={_fmt_price_debug(actual_fill_price)}; "
            f"realized_slippage_bps={_fmt_price_debug(realized_slippage_bps)}; "
            f"order_latency_ms={order_latency_ms}; "
            f"book_depth_at_fill={_fmt_price_debug(book_depth_at_fill)}; "
            f"high_confidence_size_override={used_high_confidence_override}; "
            f"required_cash={required_cash:.3f}; "
            f"estimated_fee={estimated_fee:.3f}; fee_rate_bps={cfg.live_fee_rate_bps}; "
            f"{snapshot.submit_reason})"
        ),
        live_order_response=response,
        execution_snapshot=snapshot,
        quoted_price_at_entry=quoted_price_at_entry,
        actual_fill_price=actual_fill_price,
        realized_slippage_bps=realized_slippage_bps,
        order_latency_ms=order_latency_ms,
        book_depth_at_fill=book_depth_at_fill,
        shares_requested=size,
    )


def maybe_execute_trade(
    market: BtcUpDownMarket,
    decision: LlmDecision,
    features: Optional[BtcFeatures] = None,
    snapshot: Optional[TokenQuoteSnapshot] = None,
) -> TradeExecutionResult:
    validated_snapshot, rejection = _validate_trade_candidate(
        market=market,
        decision=decision,
        features=features,
        snapshot=snapshot,
    )
    if rejection is not None:
        return rejection

    assert validated_snapshot is not None
    effective_confidence = get_effective_decision_confidence(
        decision,
        market,
        features=features,
    )

    cfg = get_trading_config()
    if cfg.paper_trading:
        return _execute_paper_trade(
            decision=decision,
            snapshot=validated_snapshot,
            effective_confidence=effective_confidence,
        )

    return _execute_live_trade(
        decision=decision,
        market=market,
        snapshot=validated_snapshot,
        effective_confidence=effective_confidence,
    )
