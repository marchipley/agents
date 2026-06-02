from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal, Optional


PositionStatus = Literal["WINNING", "LOSING", "TIED"]


@dataclass
class ActivePaperOrder:
    market_slug: str
    market_title: str
    side: str
    shares: float
    entry_price: float
    token_id: str
    target_btc_price: float
    entry_btc_price: float
    quoted_price_at_entry: Optional[float] = None
    actual_fill_price: Optional[float] = None
    realized_slippage_bps: Optional[float] = None
    order_latency_ms: Optional[int] = None
    book_depth_at_fill: Optional[float] = None
    shares_requested: Optional[float] = None
    live_order_id: Optional[str] = None
    filled: bool = False
    api_order_state: Optional[str] = None
    api_state: Optional[str] = None
    live_reprice_attempts: int = 0
    llm_prompt_text: Optional[str] = None
    llm_raw_response_text: Optional[str] = None
    trade_number_in_period: int = 1
    target_is_approximate: bool = False
    placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PaperTradingState:
    market_slug: Optional[str] = None
    market_title: Optional[str] = None
    trades_executed: int = 0
    trade_cooldown_loops_remaining: int = 0
    active_orders: List[ActivePaperOrder] = field(default_factory=list)
    realized_pnl_usd: float = 0.0
    peak_realized_pnl_usd: float = 0.0


_STATE = PaperTradingState()


def reset_period_state(market_slug: str, market_title: str) -> None:
    global _STATE
    _STATE = PaperTradingState(
        market_slug=market_slug,
        market_title=market_title,
    )


def sync_period_state(market_slug: str, market_title: str) -> bool:
    if _STATE.market_slug != market_slug:
        reset_period_state(market_slug, market_title)
        return True
    return False


def get_state() -> PaperTradingState:
    return _STATE


def record_executed_trade(order: ActivePaperOrder) -> None:
    _STATE.trades_executed += 1
    _STATE.active_orders.append(order)


def _pnl_for_order_outcome(order: ActivePaperOrder, outcome_label: str) -> Optional[float]:
    if getattr(order, "actual_fill_price", None) is None and not getattr(order, "filled", False):
        return None
    try:
        shares = float(order.shares)
        entry_price = float(order.entry_price)
    except (TypeError, ValueError):
        return None
    if outcome_label == "win":
        return shares * (1.0 - entry_price)
    if outcome_label == "loss":
        return -shares * entry_price
    return 0.0


def record_realized_pnl(delta_pnl_usd: float) -> float:
    try:
        delta = float(delta_pnl_usd)
    except (TypeError, ValueError):
        delta = 0.0
    _STATE.realized_pnl_usd += delta
    _STATE.peak_realized_pnl_usd = max(_STATE.peak_realized_pnl_usd, _STATE.realized_pnl_usd)
    return _STATE.realized_pnl_usd


def record_realized_pnl_for_order(order: ActivePaperOrder, outcome_label: str) -> Optional[float]:
    pnl = _pnl_for_order_outcome(order, outcome_label)
    if pnl is None:
        return None
    return record_realized_pnl(pnl)


def revert_executed_trade(order: ActivePaperOrder) -> None:
    """
    Removes a canceled order from the active queue and frees up the trade slot
    so the bot can attempt another entry in the same period.
    """
    if order in _STATE.active_orders:
        _STATE.active_orders.remove(order)
        _STATE.trades_executed = max(0, _STATE.trades_executed - 1)


def set_trade_cooldown(loop_count: int) -> None:
    _STATE.trade_cooldown_loops_remaining = max(int(loop_count), 0)


def get_trade_cooldown_remaining() -> int:
    return max(int(_STATE.trade_cooldown_loops_remaining), 0)


def consume_trade_cooldown_loop() -> int:
    remaining = get_trade_cooldown_remaining()
    if remaining <= 0:
        return 0
    _STATE.trade_cooldown_loops_remaining = remaining - 1
    return remaining


def get_active_orders() -> List[ActivePaperOrder]:
    return list(_STATE.active_orders)


def get_realized_pnl_snapshot() -> tuple[float, float]:
    current = float(_STATE.realized_pnl_usd)
    drawdown = max(float(_STATE.peak_realized_pnl_usd) - current, 0.0)
    return current, drawdown


def classify_position(order: ActivePaperOrder, current_btc_price: float) -> PositionStatus:
    if current_btc_price == order.target_btc_price:
        return "TIED"

    if order.side == "UP":
        return "WINNING" if current_btc_price > order.target_btc_price else "LOSING"

    return "WINNING" if current_btc_price < order.target_btc_price else "LOSING"


def describe_target(order: ActivePaperOrder) -> str:
    direction = "above" if order.side == "UP" else "below"
    qualifier = "approximately " if order.target_is_approximate else ""
    return f"BTC must finish {direction} {qualifier}{order.target_btc_price:.2f}"
