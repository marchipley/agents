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
