# custom/btc_agent/main.py

from contextlib import nullcontext
from datetime import datetime, timezone
import os
import re
import select
import time
import sys
import termios
import tty
from typing import Optional
import json

from .config import get_trading_config
from .market_lookup import (
    build_price_to_beat_debug_reports,
    fetch_btc_resolution_price_for_slug,
    find_current_btc_updown_market,
    get_btc_updown_market_by_slug,
)
from .indicators import (
    build_btc_features,
    estimate_market_window_reference_price,
    fetch_btc_spot_price,
    get_feature_readiness,
)
from .llm_decision import decide_trade, test_llm_connection
from .network import describe_proxy_configuration
from .executor import (
    AccountBalanceSnapshot,
    TokenQuoteSnapshot,
    compute_recommended_limit_price,
    compute_target_limit_price,
    evaluate_ok_to_submit,
    get_effective_decision_confidence,
    get_effective_min_confidence,
    get_submission_limit_price,
    get_account_balance_snapshot,
    get_token_quote_snapshot,
    maybe_execute_trade,
)
from .paper_state import (
    ActivePaperOrder,
    classify_position,
    consume_trade_cooldown_loop,
    describe_target,
    get_active_orders,
    get_trade_cooldown_remaining,
    get_state,
    record_executed_trade,
    set_trade_cooldown,
    sync_period_state,
)
from scripts.python.check_public_ip_indonesia import (
    check_current_public_ip_location,
    is_allowed_location,
)


_FIRST_LOOP = True
_DEBUG_WRITTEN_SLUGS = set()
_SESSION_LOSS_TRADES = 0


class QuitKeyMonitor:
    def __init__(self) -> None:
        self._fd = None
        self._saved_termios = None
        self._enabled = False

    def __enter__(self):
        try:
            if not sys.stdin.isatty():
                return self
            self._fd = sys.stdin.fileno()
            self._saved_termios = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._enabled = True
        except Exception:
            self._enabled = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._enabled and self._fd is not None and self._saved_termios is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_termios)
            except Exception:
                pass

    def poll_quit_requested(self) -> bool:
        if not self._enabled:
            return False
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return False
        try:
            pressed = os.read(self._fd, 32).decode("utf-8", errors="ignore")
        except Exception:
            return False
        return "q" in pressed.lower()


def _fmt(value):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_mmss_from_seconds(seconds: Optional[int]) -> str:
    if seconds is None:
        return "None"
    try:
        total_seconds = max(int(seconds), 0)
    except (TypeError, ValueError):
        return "None"
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def _effective_confidence(decision, market=None, features=None) -> float:
    if decision is None:
        return 0.0
    if market is None:
        cfg = get_trading_config()
        confidence = float(getattr(decision, "confidence", 0.0) or 0.0)
        return confidence if confidence >= float(getattr(cfg, "min_confidence", 0.7)) else 0.0
    effective_confidence = get_effective_decision_confidence(
        decision,
        market,
        features=features,
    )
    min_confidence = get_effective_min_confidence(
        market,
        features=features,
    )
    return effective_confidence if effective_confidence >= min_confidence else 0.0


def wait_for_next_tick_or_quit(
    interval_seconds: int,
    quit_monitor=None,
    poll_interval_seconds: float = 0.25,
) -> bool:
    deadline = time.monotonic() + max(interval_seconds, 0)
    while True:
        if quit_monitor is not None and quit_monitor.poll_quit_requested():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(poll_interval_seconds, remaining))


def has_valid_price_to_beat(value) -> bool:
    if value is None:
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    return 1000 <= numeric_value <= 1_000_000


def _extract_slug_timestamp(market_slug: str) -> str:
    match = re.search(r"btc-updown-5m-(\d+)$", market_slug or "")
    return match.group(1) if match else "unknown"


def _completed_order_log_path(market_slug: str) -> str:
    return _completed_order_log_path_for_trade(market_slug)


def _trade_number_suffix(trade_number_in_period: Optional[int]) -> str:
    cfg = get_trading_config()
    if getattr(cfg, "max_trades_per_period", 1) <= 1:
        return ""
    if not trade_number_in_period or trade_number_in_period < 1:
        return ""
    return f"-{trade_number_in_period}"


def _completed_order_log_path_for_trade(
    market_slug: str,
    trade_number_in_period: Optional[int] = None,
) -> str:
    completed_orders_dir = os.path.join(os.getcwd(), "completed_orders")
    os.makedirs(completed_orders_dir, exist_ok=True)
    return os.path.join(
        completed_orders_dir,
        f"completed_order_{_extract_slug_timestamp(market_slug)}{_trade_number_suffix(trade_number_in_period)}.txt",
    )


def _pending_period_log_path(market_slug: str) -> str:
    completed_orders_dir = os.path.join(os.getcwd(), "completed_orders")
    os.makedirs(completed_orders_dir, exist_ok=True)
    return os.path.join(
        completed_orders_dir,
        f"pending_period_{_extract_slug_timestamp(market_slug)}.txt",
    )


def _completed_period_log_path(market_slug: str) -> str:
    completed_orders_dir = os.path.join(os.getcwd(), "completed_orders")
    os.makedirs(completed_orders_dir, exist_ok=True)
    return os.path.join(
        completed_orders_dir,
        f"completed_period_{_extract_slug_timestamp(market_slug)}.txt",
    )


def _completed_period_final_log_path(
    market_slug: str,
    period_direction: str,
) -> str:
    completed_orders_dir = os.path.join(os.getcwd(), "completed_orders")
    os.makedirs(completed_orders_dir, exist_ok=True)
    return os.path.join(
        completed_orders_dir,
        f"completed_period_{period_direction}_{_extract_slug_timestamp(market_slug)}.txt",
    )


def _completed_order_final_log_path(
    market_slug: str,
    outcome_label: str,
    side: Optional[str] = None,
    trade_number_in_period: Optional[int] = None,
) -> str:
    completed_orders_dir = os.path.join(os.getcwd(), "completed_orders")
    os.makedirs(completed_orders_dir, exist_ok=True)
    side_suffix = f"_{str(side).lower()}" if side else ""
    return os.path.join(
        completed_orders_dir,
        f"completed_order_{outcome_label}{side_suffix}_{_extract_slug_timestamp(market_slug)}{_trade_number_suffix(trade_number_in_period)}.txt",
    )


def _completed_order_attempt_log_path(
    market_slug: str,
    trade_number_in_period: Optional[int] = None,
) -> str:
    completed_orders_dir = os.path.join(os.getcwd(), "completed_orders")
    os.makedirs(completed_orders_dir, exist_ok=True)
    return os.path.join(
        completed_orders_dir,
        f"completed_order_attempt_{_extract_slug_timestamp(market_slug)}{_trade_number_suffix(trade_number_in_period)}.txt",
    )


def _classify_outcome_label(position_state: str) -> str:
    if position_state == "WINNING":
        return "win"
    if position_state == "LOSING":
        return "loss"
    return "tied"


def _classify_period_direction(current_btc_price: float, target_btc_price: float) -> str:
    if current_btc_price > target_btc_price:
        return "up"
    if current_btc_price < target_btc_price:
        return "down"
    return "tied"


def _extract_period_open_price_to_beat_from_log(log_path: str) -> Optional[float]:
    try:
        with open(log_path, encoding="utf-8") as log_file:
            for line in log_file:
                if line.startswith("period_open_price_to_beat="):
                    value = line.split("=", 1)[1].strip()
                    try:
                        return float(value)
                    except ValueError:
                        return None
    except FileNotFoundError:
        return None
    return None


def _position_outcome_reason(order: ActivePaperOrder, current_btc_price: float, position_state: str) -> str:
    if order.side == "UP":
        if position_state == "WINNING":
            return (
                f"BTC finished {current_btc_price - order.target_btc_price:.2f} above the price to beat "
                f"({current_btc_price:.2f} > {order.target_btc_price:.2f})"
            )
        if position_state == "LOSING":
            return (
                f"BTC finished {order.target_btc_price - current_btc_price:.2f} below the required level "
                f"({current_btc_price:.2f} <= {order.target_btc_price:.2f})"
            )
        return f"BTC finished exactly at the price to beat ({current_btc_price:.2f})"

    if position_state == "WINNING":
        return (
            f"BTC finished {order.target_btc_price - current_btc_price:.2f} below the price to beat "
            f"({current_btc_price:.2f} < {order.target_btc_price:.2f})"
        )
    if position_state == "LOSING":
        return (
            f"BTC finished {current_btc_price - order.target_btc_price:.2f} above the required level "
            f"({current_btc_price:.2f} >= {order.target_btc_price:.2f})"
        )
    return f"BTC finished exactly at the price to beat ({current_btc_price:.2f})"


def _snapshot_summary(prefix: str, snapshot: TokenQuoteSnapshot) -> list[str]:
    return [
        f"{prefix}_buy_quote={_fmt(getattr(snapshot, 'buy_quote', None))}",
        f"{prefix}_reference_price={_fmt(getattr(snapshot, 'reference_price', None))}",
        f"{prefix}_target_limit_price={_fmt(getattr(snapshot, 'target_limit_price', None))}",
        f"{prefix}_recommended_limit_price={_fmt(getattr(snapshot, 'recommended_limit_price', None))}",
        f"{prefix}_ok_to_submit={getattr(snapshot, 'ok_to_submit', None)}",
        f"{prefix}_submit_reason={getattr(snapshot, 'submit_reason', '')}",
        f"{prefix}_best_bid={_fmt(getattr(snapshot, 'best_bid', None))}",
        f"{prefix}_best_ask={_fmt(getattr(snapshot, 'best_ask', None))}",
        f"{prefix}_best_bid_size={_fmt(getattr(snapshot, 'best_bid_size', None))}",
        f"{prefix}_best_ask_size={_fmt(getattr(snapshot, 'best_ask_size', None))}",
        f"{prefix}_spread={_fmt(getattr(snapshot, 'spread', None))}",
        f"{prefix}_spread_bps={_fmt(getattr(snapshot, 'spread_bps', None))}",
        f"{prefix}_top_level_book_imbalance={_fmt(getattr(snapshot, 'top_level_book_imbalance', None))}",
        f"{prefix}_imbalance_pressure={_fmt(getattr(snapshot, 'imbalance_pressure', None))}",
    ]


def _slug_start_ts(market_slug: str) -> Optional[int]:
    try:
        return int(_extract_slug_timestamp(market_slug))
    except (TypeError, ValueError):
        return None


def _market_time_remaining_seconds(market_slug: str, observed_at: datetime) -> Optional[int]:
    start_ts = _slug_start_ts(market_slug)
    if start_ts is None:
        return None
    if not hasattr(observed_at, "timestamp"):
        return None
    end_ts = start_ts + 300
    return max(end_ts - int(observed_at.timestamp()), 0)


def _volatility_regime(volatility_5m) -> str:
    if volatility_5m is None:
        return "unknown"
    if volatility_5m < 5:
        return "low"
    if volatility_5m < 12:
        return "medium"
    if volatility_5m < 25:
        return "high"
    return "extreme"


def _trend_regime(features, period_open_price_to_beat: Optional[float] = None) -> str:
    delta_pct = getattr(features, "delta_pct_from_window_open", None)
    momentum_5m = getattr(features, "momentum_5m", None)
    price_usd = getattr(features, "price_usd", None)
    if delta_pct is None or momentum_5m is None:
        return "unknown"

    if price_usd not in (None, 0) and period_open_price_to_beat not in (None, 0):
        gap_pct = (price_usd - period_open_price_to_beat) / period_open_price_to_beat
        if gap_pct > 0.0005 and not (delta_pct < -0.0015 and momentum_5m < -15):
            if gap_pct > 0.0015 or delta_pct > 0.0015 or momentum_5m > 15:
                return "strong_up"
            return "weak_up"
        if gap_pct < -0.0005 and not (delta_pct > 0.0015 and momentum_5m > 15):
            if gap_pct < -0.0015 or delta_pct < -0.0015 or momentum_5m < -15:
                return "strong_down"
            return "weak_down"

    if abs(delta_pct) < 0.0005 and abs(momentum_5m) < 8:
        return "ranging"
    if delta_pct > 0.0015 and momentum_5m > 15:
        return "strong_up"
    if delta_pct < -0.0015 and momentum_5m < -15:
        return "strong_down"
    if delta_pct > 0:
        return "weak_up"
    if delta_pct < 0:
        return "weak_down"
    return "mixed"


def _liquidity_regime(snapshot: TokenQuoteSnapshot) -> str:
    spread_bps = getattr(snapshot, "spread_bps", None)
    best_bid_size = getattr(snapshot, "best_bid_size", None)
    best_ask_size = getattr(snapshot, "best_ask_size", None)
    imbalance_pressure = getattr(snapshot, "imbalance_pressure", None)
    total_top_size = 0.0
    for size in (best_bid_size, best_ask_size):
        if size is not None:
            total_top_size += size
    if spread_bps is None and total_top_size <= 0:
        return "unknown"
    if spread_bps is not None and spread_bps > 150:
        return "THIN_LIQUIDITY"
    if spread_bps is not None and spread_bps > 80:
        return "low"
    if total_top_size > 0 and total_top_size < 25:
        return "low"
    if imbalance_pressure is not None and abs(imbalance_pressure) > 0.60:
        return "thin_imbalanced"
    if spread_bps is None:
        if total_top_size >= 100:
            return "high"
        if total_top_size > 0:
            return "normal"
        return "unknown"
    if spread_bps <= 30:
        return "high"
    if spread_bps <= 80:
        return "normal"
    return "low"


def _rsi_regime(features) -> str:
    rsi_fast = getattr(features, "rsi_9", None)
    rsi_slow = getattr(features, "rsi_14", None)
    rsi = rsi_fast if rsi_fast is not None else rsi_slow
    if rsi is None:
        return "unknown"
    momentum_5m = getattr(features, "momentum_5m", None)
    delta_pct = getattr(features, "delta_pct_from_window_open", None)
    price_usd = getattr(features, "price_usd", None)
    parabolic_threshold = None
    if momentum_5m is not None and price_usd not in (None, 0):
        parabolic_threshold = abs(momentum_5m) / price_usd
    if (
        rsi_fast is not None
        and rsi_fast >= 80
        and momentum_5m is not None
        and momentum_5m > 0
        and parabolic_threshold is not None
        and parabolic_threshold > 0.0003
    ):
        return "PARABOLIC_UP"
    if (
        rsi_fast is not None
        and rsi_fast <= 20
        and momentum_5m is not None
        and momentum_5m < 0
        and parabolic_threshold is not None
        and parabolic_threshold > 0.0003
    ):
        return "PARABOLIC_DOWN"
    if rsi >= 80:
        if (momentum_5m is not None and momentum_5m > 15) or (delta_pct is not None and delta_pct > 0.0015):
            return "strong_trend_high"
        return "extreme_overbought"
    if rsi >= 70:
        if (momentum_5m is not None and momentum_5m > 8) or (delta_pct is not None and delta_pct > 0.0008):
            return "trend_high"
        return "overbought"
    if rsi <= 20:
        if (momentum_5m is not None and momentum_5m < -15) or (delta_pct is not None and delta_pct < -0.0015):
            return "strong_trend_low"
        return "extreme_oversold"
    if rsi <= 30:
        if (momentum_5m is not None and momentum_5m < -8) or (delta_pct is not None and delta_pct < -0.0008):
            return "trend_low"
        return "oversold"
    return "neutral"


def _build_regime_fingerprint(
    *,
    market=None,
    market_slug: str,
    observed_at: datetime,
    features=None,
    up_snapshot: TokenQuoteSnapshot = None,
    down_snapshot: TokenQuoteSnapshot = None,
    current_btc_price: Optional[float] = None,
    period_open_price_to_beat: Optional[float] = None,
) -> dict:
    current_price = current_btc_price
    if current_price is None and features is not None:
        current_price = getattr(features, "price_usd", None)

    gap_to_target = None
    gap_to_target_pct = None
    strike_delta_pct = None
    time_remaining_seconds = _market_time_remaining_seconds(market_slug, observed_at)
    if current_price is not None and period_open_price_to_beat not in (None, 0):
        gap_to_target = current_price - period_open_price_to_beat
        gap_to_target_pct = (gap_to_target / period_open_price_to_beat) * 100
        if current_price != 0:
            strike_delta_pct = (gap_to_target / current_price) * 100

    required_velocity_to_win = None
    if gap_to_target is not None and time_remaining_seconds not in (None, 0):
        required_velocity_to_win = abs(gap_to_target) / time_remaining_seconds
    atr_14 = None if features is None else getattr(features, "atr_14", None)
    volatility_normalized_gap = None
    if gap_to_target is not None and atr_14 not in (None, 0):
        volatility_normalized_gap = abs(gap_to_target) / atr_14
    oracle_gap_ratio = None
    if gap_to_target is not None and atr_14 not in (None, 0):
        oracle_gap_ratio = gap_to_target / atr_14
    implied_oracle_price = None
    feed_drift_usd = None
    if (
        period_open_price_to_beat not in (None, 0)
        and atr_14 not in (None, 0)
        and up_snapshot is not None
        and down_snapshot is not None
        and getattr(up_snapshot, "buy_quote", None) is not None
        and getattr(down_snapshot, "buy_quote", None) is not None
    ):
        implied_oracle_price = (
            period_open_price_to_beat
            + (float(up_snapshot.buy_quote) - float(down_snapshot.buy_quote)) * atr_14
        )
        if current_price is not None:
            feed_drift_usd = current_price - implied_oracle_price

    selected_snapshot = None
    if up_snapshot is not None and down_snapshot is not None:
        up_spread_bps = getattr(up_snapshot, "spread_bps", None)
        down_spread_bps = getattr(down_snapshot, "spread_bps", None)
        if up_spread_bps is not None and down_spread_bps is not None:
            selected_snapshot = up_snapshot if up_spread_bps <= down_spread_bps else down_snapshot
        else:
            selected_snapshot = up_snapshot
    else:
        selected_snapshot = up_snapshot or down_snapshot

    return {
        "time_remaining_seconds": time_remaining_seconds,
        "next_slug_proximity": time_remaining_seconds,
        "volatility_regime": _volatility_regime(getattr(features, "volatility_5m", None)) if features is not None else "unknown",
        "trend_regime": _trend_regime(features, period_open_price_to_beat=period_open_price_to_beat) if features is not None else "unknown",
        "rsi_regime": _rsi_regime(features) if features is not None else "unknown",
        "liquidity_regime": _liquidity_regime(selected_snapshot) if selected_snapshot is not None else "unknown",
        "threshold_gap_usd": None if gap_to_target is None else round(gap_to_target, 4),
        "threshold_gap_pct": None if gap_to_target_pct is None else round(gap_to_target_pct, 4),
        "strike_delta_pct": None if strike_delta_pct is None else round(strike_delta_pct, 4),
        "up_market_probability": None if market is None else getattr(market, "up_market_probability", None),
        "down_market_probability": None if market is None else getattr(market, "down_market_probability", None),
        "required_velocity_to_win": None
        if required_velocity_to_win is None
        else round(required_velocity_to_win, 4),
        "oracle_gap_ratio": None
        if oracle_gap_ratio is None
        else round(oracle_gap_ratio, 4),
        "implied_oracle_price": None
        if implied_oracle_price is None
        else round(implied_oracle_price, 4),
        "feed_drift_usd": None
        if feed_drift_usd is None
        else round(feed_drift_usd, 4),
        "rsi_speed_divergence": None
        if features is None
        else getattr(features, "rsi_speed_divergence", None),
        "trend_intensity": None
        if features is None
        else getattr(features, "adx_14", None),
        "ema_alignment": None
        if features is None
        else getattr(features, "ema_alignment", None),
        "volatility_normalized_gap": None
        if volatility_normalized_gap is None
        else round(volatility_normalized_gap, 4),
        "window_delta_pct": None
        if features is None or getattr(features, "delta_pct_from_window_open", None) is None
        else round(getattr(features, "delta_pct_from_window_open") * 100, 4),
        "top_level_book_imbalance": None
        if selected_snapshot is None
        else getattr(selected_snapshot, "top_level_book_imbalance", None),
        "imbalance_pressure": None
        if selected_snapshot is None
        else getattr(selected_snapshot, "imbalance_pressure", None),
        "velocity_15s": None
        if features is None
        else getattr(features, "velocity_15s", None),
        "velocity_30s": None
        if features is None
        else getattr(features, "velocity_30s", None),
        "consecutive_flat_ticks": None
        if features is None
        else getattr(features, "consecutive_flat_ticks", None),
        "consecutive_directional_ticks": None
        if features is None
        else getattr(features, "consecutive_directional_ticks", None),
        "last_10_ticks_direction": None
        if features is None
        else getattr(features, "last_10_ticks_direction", None),
    }


def append_pending_period_tick_analysis(
    market,
    *,
    up_snapshot: TokenQuoteSnapshot = None,
    down_snapshot: TokenQuoteSnapshot = None,
    features=None,
    decision=None,
    skip_reason: str = "",
    observed_at: datetime = None,
) -> None:
    observed_at = observed_at or datetime.now(timezone.utc)
    log_path = _pending_period_log_path(market.slug)
    with open(log_path, "a", encoding="utf-8") as log_file:
        if log_file.tell() == 0:
            log_file.write(
                "\n".join(
                    [
                        f"market_slug={market.slug}",
                        f"market_title={market.title}",
                        f"slug_timestamp={_extract_slug_timestamp(market.slug)}",
                        "",
                    ]
                )
            )

        lines = [
            f"observed_at={observed_at.isoformat()}",
            "phase=PRE_ORDER_TICK",
            f"period_open_price_to_beat={_fmt(market.settlement_threshold)}",
            f"up_market_probability={_fmt(getattr(market, 'up_market_probability', None))}",
            f"down_market_probability={_fmt(getattr(market, 'down_market_probability', None))}",
            f"market_time_remaining_seconds={_fmt(_market_time_remaining_seconds(market.slug, observed_at))}",
            f"market_time_remaining_mmss={_fmt_mmss_from_seconds(_market_time_remaining_seconds(market.slug, observed_at))}",
        ]
        if up_snapshot is not None:
            lines.extend(_snapshot_summary("up", up_snapshot))
        if down_snapshot is not None:
            lines.extend(_snapshot_summary("down", down_snapshot))

        if features is not None:
            lines.extend(
                [
                    f"btc_price={_fmt(getattr(features, 'price_usd', None))}",
                    f"delta_prev_tick={getattr(features, 'delta_from_previous_tick', None)}",
                    f"momentum_1m={getattr(features, 'momentum_1m', None)}",
                    f"momentum_5m={getattr(features, 'momentum_5m', None)}",
                    f"velocity_15s={getattr(features, 'velocity_15s', None)}",
                    f"velocity_30s={getattr(features, 'velocity_30s', None)}",
                    f"momentum_acceleration={getattr(features, 'momentum_acceleration', None)}",
                    f"volatility_5m={getattr(features, 'volatility_5m', None)}",
                    f"consecutive_flat_ticks={getattr(features, 'consecutive_flat_ticks', None)}",
                    f"consecutive_directional_ticks={getattr(features, 'consecutive_directional_ticks', None)}",
                    f"last_10_ticks_direction={getattr(features, 'last_10_ticks_direction', None)}",
                    f"rsi_9={getattr(features, 'rsi_9', None)}",
                    f"rsi_speed_divergence={getattr(features, 'rsi_speed_divergence', None)}",
                    f"ema_9={getattr(features, 'ema_9', None)}",
                    f"ema_21={getattr(features, 'ema_21', None)}",
                    f"ema_alignment={getattr(features, 'ema_alignment', None)}",
                    f"ema_cross_direction={getattr(features, 'ema_cross_direction', None)}",
                    f"adx_14={getattr(features, 'adx_14', None)}",
                    f"atr_14={getattr(features, 'atr_14', None)}",
                    f"window_open_price={_fmt(getattr(features, 'window_open_price', None))}",
                    f"delta_from_window_pct={((getattr(features, 'delta_pct_from_window_open', 0.0) or 0.0) * 100):.4f}%",
                    f"trailing_5m_open_price={_fmt(getattr(features, 'trailing_5m_open_price', None))}",
                    f"delta_from_5m_pct={((getattr(features, 'delta_pct_from_trailing_5m_open', 0.0) or 0.0) * 100):.4f}%",
                    f"rsi_14={getattr(features, 'rsi_14', None)}",
                ]
            )
            lines.append(
                "regime_fingerprint="
                + json.dumps(
                    _build_regime_fingerprint(
                        market_slug=market.slug,
                        market=market,
                        observed_at=observed_at,
                        features=features,
                        up_snapshot=up_snapshot,
                        down_snapshot=down_snapshot,
                        period_open_price_to_beat=market.settlement_threshold,
                    ),
                    sort_keys=True,
                )
            )

        if decision is not None:
            lines.extend(
                [
                    f"decision_side={decision.side}",
                    f"decision_confidence={decision.confidence:.3f}",
                    f"effective_confidence={_effective_confidence(decision, market=market, features=features):.3f}",
                    f"decision_max_price_to_pay={decision.max_price_to_pay:.3f}",
                    f"decision_reason={decision.reason}",
                ]
            )
            prompt_text = getattr(decision, "prompt_text", None)
            if prompt_text:
                lines.extend(
                    [
                        "llm_prompt_start",
                        prompt_text,
                        "llm_prompt_end",
                    ]
                )

        if skip_reason:
            lines.append(f"skip_reason={skip_reason}")

        lines.append("")
        log_file.write("\n".join(lines))


def promote_pending_period_log_to_completed(
    market_slug: str,
    trade_number_in_period: Optional[int] = None,
) -> None:
    pending_path = _pending_period_log_path(market_slug)
    completed_path = _completed_order_log_path_for_trade(market_slug, trade_number_in_period)
    if not os.path.exists(pending_path):
        return
    if os.path.exists(completed_path):
        return
    with open(pending_path, "r", encoding="utf-8") as pending_file:
        content = pending_file.read()
    with open(completed_path, "w", encoding="utf-8") as completed_file:
        completed_file.write(content)


def finalize_pending_period_log(market_slug: str, final_btc_price: Optional[float] = None) -> None:
    if not market_slug:
        return
    pending_path = _pending_period_log_path(market_slug)
    if not os.path.exists(pending_path):
        return
    completed_path = _completed_period_log_path(market_slug)
    if final_btc_price is not None:
        target_btc_price = _extract_period_open_price_to_beat_from_log(pending_path)
        if target_btc_price not in (None, 0):
            completed_path = _completed_period_final_log_path(
                market_slug,
                _classify_period_direction(final_btc_price, target_btc_price),
            )
    if os.path.exists(completed_path):
        try:
            os.remove(pending_path)
        except FileNotFoundError:
            pass
        return
    os.replace(pending_path, completed_path)


def append_completed_order_tick(
    order: ActivePaperOrder,
    current_btc_price: float,
    phase: str,
    observed_at: datetime = None,
    features=None,
    up_snapshot: TokenQuoteSnapshot = None,
    down_snapshot: TokenQuoteSnapshot = None,
) -> None:
    observed_at = observed_at or datetime.now(timezone.utc)
    log_path = _completed_order_log_path_for_trade(
        order.market_slug,
        getattr(order, "trade_number_in_period", None),
    )
    status = classify_position(order, current_btc_price)
    btc_move_from_entry = current_btc_price - order.entry_btc_price
    btc_move_from_entry_pct = (
        (btc_move_from_entry / order.entry_btc_price) * 100
        if order.entry_btc_price
        else 0.0
    )
    btc_gap_to_target = current_btc_price - order.target_btc_price
    outcome_label = _classify_outcome_label(status)
    outcome_reason = _position_outcome_reason(order, current_btc_price, status)
    regime_fingerprint = _build_regime_fingerprint(
        market_slug=order.market_slug,
        observed_at=observed_at,
        features=features,
        up_snapshot=up_snapshot,
        down_snapshot=down_snapshot,
        current_btc_price=current_btc_price,
        period_open_price_to_beat=order.target_btc_price,
    )
    with open(log_path, "a", encoding="utf-8") as log_file:
        if log_file.tell() == 0:
            log_file.write(
                "\n".join(
                    [
                        f"market_slug={order.market_slug}",
                        f"market_title={order.market_title}",
                        f"slug_timestamp={_extract_slug_timestamp(order.market_slug)}",
                        "",
                    ]
                )
            )
        log_file.write(
            "\n".join(
                [
                    f"observed_at={observed_at.isoformat()}",
                    f"phase={phase}",
                    f"side={order.side}",
                    f"shares={order.shares:.4f}",
                    f"entry_price={order.entry_price:.3f}",
                    f"entry_btc_price={order.entry_btc_price:.2f}",
                    f"period_open_price_to_beat={order.target_btc_price:.2f}",
                    f"current_btc_price={current_btc_price:.2f}",
                    f"btc_move_from_entry={btc_move_from_entry:.2f}",
                    f"btc_move_from_entry_pct={btc_move_from_entry_pct:.4f}%",
                    f"btc_gap_to_target={btc_gap_to_target:.2f}",
                    f"market_time_remaining_seconds={_fmt(regime_fingerprint.get('time_remaining_seconds'))}",
                    f"market_time_remaining_mmss={_fmt_mmss_from_seconds(regime_fingerprint.get('time_remaining_seconds'))}",
                    f"position_state={status}",
                    f"target_description={describe_target(order)}",
                    f"outcome_label={outcome_label}",
                    f"outcome_reason={outcome_reason}",
                    "",
                ]
            )
        )
        if features is not None:
            log_file.write(
                "\n".join(
                    [
                        f"feature_btc_price={_fmt(getattr(features, 'price_usd', None))}",
                        f"feature_delta_prev_tick={getattr(features, 'delta_from_previous_tick', None)}",
                        f"feature_momentum_1m={getattr(features, 'momentum_1m', None)}",
                        f"feature_momentum_5m={getattr(features, 'momentum_5m', None)}",
                        f"feature_velocity_15s={getattr(features, 'velocity_15s', None)}",
                        f"feature_velocity_30s={getattr(features, 'velocity_30s', None)}",
                        f"feature_momentum_acceleration={getattr(features, 'momentum_acceleration', None)}",
                        f"feature_volatility_5m={getattr(features, 'volatility_5m', None)}",
                        f"feature_consecutive_flat_ticks={getattr(features, 'consecutive_flat_ticks', None)}",
                        f"feature_consecutive_directional_ticks={getattr(features, 'consecutive_directional_ticks', None)}",
                        f"feature_last_10_ticks_direction={getattr(features, 'last_10_ticks_direction', None)}",
                        f"feature_rsi_9={getattr(features, 'rsi_9', None)}",
                        f"feature_rsi_speed_divergence={getattr(features, 'rsi_speed_divergence', None)}",
                        f"feature_ema_9={getattr(features, 'ema_9', None)}",
                        f"feature_ema_21={getattr(features, 'ema_21', None)}",
                        f"feature_ema_alignment={getattr(features, 'ema_alignment', None)}",
                        f"feature_ema_cross_direction={getattr(features, 'ema_cross_direction', None)}",
                        f"feature_adx_14={getattr(features, 'adx_14', None)}",
                        f"feature_atr_14={getattr(features, 'atr_14', None)}",
                        f"feature_window_open_price={_fmt(getattr(features, 'window_open_price', None))}",
                        f"feature_delta_from_window_pct={((getattr(features, 'delta_pct_from_window_open', 0.0) or 0.0) * 100):.4f}%",
                        f"feature_trailing_5m_open_price={_fmt(getattr(features, 'trailing_5m_open_price', None))}",
                        f"feature_delta_from_5m_pct={((getattr(features, 'delta_pct_from_trailing_5m_open', 0.0) or 0.0) * 100):.4f}%",
                        f"feature_rsi_14={getattr(features, 'rsi_14', None)}",
                        "regime_fingerprint=" + json.dumps(regime_fingerprint, sort_keys=True),
                        "",
                    ]
                )
            )
        if up_snapshot is not None:
            log_file.write("\n".join(_snapshot_summary("active_up", up_snapshot) + [""]))
        if down_snapshot is not None:
            log_file.write("\n".join(_snapshot_summary("active_down", down_snapshot) + [""]))
    if phase == "COMPLETED":
        period_direction = _classify_period_direction(current_btc_price, order.target_btc_price)
        final_path = _completed_order_final_log_path(
            order.market_slug,
            outcome_label,
            period_direction,
            getattr(order, "trade_number_in_period", None),
        )
        try:
            os.replace(log_path, final_path)
        except FileNotFoundError:
            pass
    return outcome_label


def finalize_completed_orders(previous_orders, current_btc_price: float) -> int:
    loss_count = 0
    for order in previous_orders:
        outcome_label = append_completed_order_tick(
            order,
            current_btc_price=current_btc_price,
            phase="COMPLETED",
        )
        if outcome_label == "loss":
            loss_count += 1
    return loss_count


def finalize_current_period_logs_on_exit() -> None:
    state = get_state()
    market_slug = getattr(state, "market_slug", None)
    if market_slug:
        finalize_pending_period_log(market_slug)


def update_active_order_logs(
    current_btc_price: float,
    observed_at: datetime = None,
    features=None,
    up_snapshot: TokenQuoteSnapshot = None,
    down_snapshot: TokenQuoteSnapshot = None,
) -> None:
    for order in get_active_orders():
        append_completed_order_tick(
            order,
            current_btc_price=current_btc_price,
            phase="ACTIVE",
            observed_at=observed_at,
            features=features,
            up_snapshot=up_snapshot,
            down_snapshot=down_snapshot,
        )


def append_failed_order_attempt(
    market,
    decision,
    result,
    *,
    features=None,
    up_snapshot: TokenQuoteSnapshot = None,
    down_snapshot: TokenQuoteSnapshot = None,
    observed_at: datetime = None,
    trade_number_in_period: Optional[int] = None,
) -> None:
    observed_at = observed_at or datetime.now(timezone.utc)
    log_path = _completed_order_attempt_log_path(
        market.slug,
        trade_number_in_period=trade_number_in_period,
    )
    regime_fingerprint = _build_regime_fingerprint(
        market_slug=market.slug,
        observed_at=observed_at,
        features=features,
        up_snapshot=up_snapshot,
        down_snapshot=down_snapshot,
        current_btc_price=None if features is None else getattr(features, "price_usd", None),
        period_open_price_to_beat=market.settlement_threshold,
    )
    with open(log_path, "a", encoding="utf-8") as log_file:
        if log_file.tell() == 0:
            log_file.write(
                "\n".join(
                    [
                        f"market_slug={market.slug}",
                        f"market_title={market.title}",
                        f"slug_timestamp={_extract_slug_timestamp(market.slug)}",
                        "",
                    ]
                )
            )
        lines = [
            f"observed_at={observed_at.isoformat()}",
            "phase=ATTEMPT_FAILED",
            f"period_open_price_to_beat={_fmt(market.settlement_threshold)}",
            f"attempt_side={getattr(decision, 'side', None)}",
            f"attempt_confidence={_fmt(getattr(decision, 'confidence', None))}",
            f"attempt_max_price_to_pay={_fmt(getattr(decision, 'max_price_to_pay', None))}",
            f"attempted_price={_fmt(getattr(result, 'price', None))}",
            f"attempted_size={_fmt(getattr(result, 'size', None))}",
            f"attempt_token_id={getattr(result, 'token_id', None)}",
            f"attempt_reason={getattr(result, 'reason', '')}",
        ]
        if up_snapshot is not None:
            lines.extend(_snapshot_summary("up", up_snapshot))
        if down_snapshot is not None:
            lines.extend(_snapshot_summary("down", down_snapshot))
        if features is not None:
            lines.extend(
                [
                    f"btc_price={_fmt(getattr(features, 'price_usd', None))}",
                    f"delta_prev_tick={getattr(features, 'delta_from_previous_tick', None)}",
                    f"momentum_1m={getattr(features, 'momentum_1m', None)}",
                    f"momentum_5m={getattr(features, 'momentum_5m', None)}",
                    f"velocity_15s={getattr(features, 'velocity_15s', None)}",
                    f"velocity_30s={getattr(features, 'velocity_30s', None)}",
                    f"momentum_acceleration={getattr(features, 'momentum_acceleration', None)}",
                    f"volatility_5m={getattr(features, 'volatility_5m', None)}",
                    f"consecutive_flat_ticks={getattr(features, 'consecutive_flat_ticks', None)}",
                    f"consecutive_directional_ticks={getattr(features, 'consecutive_directional_ticks', None)}",
                    f"rsi_9={getattr(features, 'rsi_9', None)}",
                    f"rsi_14={getattr(features, 'rsi_14', None)}",
                    f"rsi_speed_divergence={getattr(features, 'rsi_speed_divergence', None)}",
                    f"ema_9={getattr(features, 'ema_9', None)}",
                    f"ema_21={getattr(features, 'ema_21', None)}",
                    f"ema_alignment={getattr(features, 'ema_alignment', None)}",
                    f"ema_cross_direction={getattr(features, 'ema_cross_direction', None)}",
                    f"adx_14={getattr(features, 'adx_14', None)}",
                    f"atr_14={getattr(features, 'atr_14', None)}",
                    "regime_fingerprint=" + json.dumps(regime_fingerprint, sort_keys=True),
                ]
            )
        lines.append("")
        log_file.write("\n".join(lines))


def enforce_session_loss_trade_limit(cfg) -> None:
    if getattr(cfg, "max_automated_loss_trades", 0) <= 0:
        return
    if _SESSION_LOSS_TRADES < cfg.max_automated_loss_trades:
        return
    print(
        "Max automated loss trades for this session has been reached "
        f"({_SESSION_LOSS_TRADES}/{cfg.max_automated_loss_trades}). "
        "Exiting BTC agent."
    )
    sys.exit(0)


def _get_losing_active_orders(current_btc_price: float) -> list[ActivePaperOrder]:
    return [
        order
        for order in get_active_orders()
        if classify_position(order, current_btc_price) == "LOSING"
    ]


def write_price_to_beat_debug_file(slug: str, force: bool = False) -> None:
    if not force and slug in _DEBUG_WRITTEN_SLUGS:
        return

    try:
        reports = build_price_to_beat_debug_reports(slug)
        for index, report in enumerate(reports, start=1):
            if index == 1:
                debug_path = os.path.join(os.getcwd(), "logs", "priceToBeatDebug.txt")
                print(f"price_to_beat_debug_file: {debug_path}")
            else:
                debug_path = os.path.join(
                    os.getcwd(),
                    "logs",
                    f"priceToBeatDebugPg{index}.txt",
                )
                print(f"price_to_beat_debug_file_pg{index}: {debug_path}")
            with open(debug_path, "w", encoding="utf-8") as debug_file:
                debug_file.write(report)
        _DEBUG_WRITTEN_SLUGS.add(slug)
    except Exception as exc:
        print(f"price_to_beat_debug_file_error: {exc}")


def clear_price_to_beat_debug_files() -> None:
    logs_dir = os.path.join(os.getcwd(), "logs")
    try:
        for name in os.listdir(logs_dir):
            if not (
                name == "priceToBeatDebug.txt"
                or (name.startswith("priceToBeatDebugPg") and name.endswith(".txt"))
            ):
                continue
            try:
                os.remove(os.path.join(logs_dir, name))
            except FileNotFoundError:
                pass
    except FileNotFoundError:
        return


def resolve_price_to_beat_with_retries(market, retry_attempts: int = 2, retry_delay_seconds: int = 3):
    if has_valid_price_to_beat(market.settlement_threshold):
        return market

    cfg = get_trading_config()
    if cfg.debug_price_to_beat:
        print("price_to_beat_retry: skipped because DEBUG_PRICE_TO_BEAT=true")
        return market

    for attempt in range(1, retry_attempts + 1):
        print(
            "price_to_beat_retry: "
            f"attempt {attempt}/{retry_attempts} for {market.slug} after {retry_delay_seconds}s"
        )
        time.sleep(retry_delay_seconds)
        refreshed_market = get_btc_updown_market_by_slug(market.slug)
        if refreshed_market is not None:
            market = refreshed_market
        if has_valid_price_to_beat(market.settlement_threshold):
            return market

    return market


def print_ip_location(public_ip, location, debug: bool) -> None:
    print(f"public_ip: {public_ip or 'unknown'}")
    print(f"is_allowed_location: {str(is_allowed_location(location)).lower()}")
    print(f"lookup_success: {str(bool(location.get('success', False))).lower()}")
    print(f"country: {location.get('country', 'unknown')}")

    message = location.get("message")
    if message:
        print(f"message: {message}")

    if not debug:
        return

    print(f"country_code: {location.get('country_code', 'unknown')}")
    print(f"region: {location.get('region', 'unknown')}")
    print(f"city: {location.get('city', 'unknown')}")
    print(f"continent: {location.get('continent', 'unknown')}")
    print(f"latitude: {location.get('latitude', 'unknown')}")
    print(f"longitude: {location.get('longitude', 'unknown')}")
    print(f"asn: {location.get('connection', {}).get('asn', 'unknown')}")
    print(f"org: {location.get('connection', {}).get('org', 'unknown')}")


def print_quote_snapshot(label: str, token_id: str, decision=None, debug: bool = True) -> None:
    q = get_token_quote_snapshot(token_id, decision=decision)
    print_quote_snapshot_from_snapshot(label, q, debug=debug)


def print_quote_snapshot_from_snapshot(
    label: str,
    q: TokenQuoteSnapshot,
    debug: bool = True,
) -> None:
    cfg = get_trading_config()
    use_recommended_limit = getattr(cfg, "use_recommended_limit", True)
    print(f"{label} quote snapshot:")
    if not debug:
        if use_recommended_limit:
            print(f"  buy_quote              = {_fmt(q.buy_quote)}")
            print(f"  recommended_limit_price= {_fmt(q.recommended_limit_price)}")
        else:
            print(f"  target_limit_price     = {_fmt(q.target_limit_price)}")
            print(f"  reference_price        = {_fmt(q.reference_price)}")
        print(f"  ok_to_submit           = {q.ok_to_submit}")
        print(f"  submit_reason          = {q.submit_reason}")
        return

    print(f"  token_id               = {q.token_id}")
    print(f"  buy_quote              = {_fmt(q.buy_quote)}")
    print(f"  midpoint               = {_fmt(q.midpoint)}")
    print(f"  last_trade_price       = {_fmt(q.last_trade_price)}")
    print(f"  reference_price        = {_fmt(q.reference_price)}")
    print(f"  target_limit_price     = {_fmt(q.target_limit_price)}")
    print(f"  recommended_limit_price= {_fmt(q.recommended_limit_price)}")
    print(f"  ok_to_submit           = {q.ok_to_submit}")
    print(f"  submit_reason          = {q.submit_reason}")
    print(f"  best_bid               = {_fmt(q.best_bid)}")
    print(f"  best_ask               = {_fmt(q.best_ask)}")
    print(f"  tick_size              = {_fmt(q.tick_size)}")
    print(f"  spread                 = {_fmt(q.spread)}")


def get_decision_quote_snapshot(
    market,
    decision,
    up_snapshot: TokenQuoteSnapshot,
    down_snapshot: TokenQuoteSnapshot,
) -> TokenQuoteSnapshot:
    base_snapshot = up_snapshot if decision.side == "UP" else down_snapshot
    target_limit_price = compute_target_limit_price(
        base_snapshot.reference_price,
        decision=decision,
    )
    recommended_limit_price = compute_recommended_limit_price(
        base_snapshot.reference_price,
        base_snapshot.tick_size,
        decision=decision,
    )
    provisional_snapshot = TokenQuoteSnapshot(
        token_id=base_snapshot.token_id,
        buy_quote=base_snapshot.buy_quote,
        midpoint=base_snapshot.midpoint,
        last_trade_price=base_snapshot.last_trade_price,
        reference_price=base_snapshot.reference_price,
        target_limit_price=target_limit_price,
        recommended_limit_price=recommended_limit_price,
        ok_to_submit=False,
        submit_reason="",
        best_bid=base_snapshot.best_bid,
        best_ask=base_snapshot.best_ask,
        tick_size=base_snapshot.tick_size,
        spread=base_snapshot.spread,
    )

    ok_to_submit, submit_reason = evaluate_ok_to_submit(
        buy_quote=base_snapshot.buy_quote,
        reference_price=base_snapshot.reference_price,
        submission_limit_price=get_submission_limit_price(provisional_snapshot),
        tick_size=base_snapshot.tick_size,
    )
    return TokenQuoteSnapshot(
        token_id=base_snapshot.token_id,
        buy_quote=base_snapshot.buy_quote,
        midpoint=base_snapshot.midpoint,
        last_trade_price=base_snapshot.last_trade_price,
        reference_price=base_snapshot.reference_price,
        target_limit_price=target_limit_price,
        recommended_limit_price=recommended_limit_price,
        ok_to_submit=ok_to_submit,
        submit_reason=submit_reason,
        best_bid=base_snapshot.best_bid,
        best_ask=base_snapshot.best_ask,
        tick_size=base_snapshot.tick_size,
        spread=base_snapshot.spread,
    )


def print_account_snapshot(debug: bool) -> None:
    account = get_account_balance_snapshot()
    print_account_snapshot_from_snapshot(account, debug=debug)


def print_account_snapshot_from_snapshot(account: AccountBalanceSnapshot, debug: bool) -> None:
    print("Account balances:")
    if not debug:
        print(f"  cash_balance_pusd      = {_fmt(account.cash_balance)}")
        print(f"  legacy_usdc_balance    = {_fmt(account.legacy_usdc_balance)}")
        print(f"  portfolio_balance_usd  = {_fmt(account.portfolio_balance)}")
        print(f"  total_account_value_usd= {_fmt(account.total_account_value)}")
        return

    print(f"  signer_address         = {account.signer_address}")
    print(f"  balance_address        = {account.balance_address}")
    print(f"  proxy_address          = {account.proxy_address or 'None'}")
    print(f"  cash_balance_pusd      = {_fmt(account.cash_balance)}")
    print(f"  legacy_usdc_balance    = {_fmt(account.legacy_usdc_balance)}")
    print(f"  portfolio_balance_usd  = {_fmt(account.portfolio_balance)}")
    print(f"  total_account_value_usd= {_fmt(account.total_account_value)}")
    print(f"  balance_error          = {account.error or 'None'}")


def enforce_minimum_wallet_balance(account: AccountBalanceSnapshot) -> None:
    cfg = get_trading_config()
    if cfg.minimum_wallet_balance <= 0:
        return
    if account.cash_balance is None:
        print(
            "ERROR: Unable to verify cash_balance_pusd for MINIMUM_WALLET_BALANCE "
            f"check ({cfg.minimum_wallet_balance:.3f}). Aborting BTC agent startup."
        )
        sys.exit(1)
    if account.cash_balance < cfg.minimum_wallet_balance:
        print(
            "ERROR: Nothing can be executed because cash_balance_pusd is below "
            "MINIMUM_WALLET_BALANCE "
            f"(available={account.cash_balance:.3f}, "
            f"minimum={cfg.minimum_wallet_balance:.3f})."
        )
        sys.exit(1)


def print_active_orders(current_btc_price: float) -> None:
    active_orders = get_active_orders()
    if not active_orders:
        print("Active orders: None")
        return

    print("Active orders:")
    for idx, order in enumerate(active_orders, start=1):
        status = classify_position(order, current_btc_price)
        win_condition = (
            f"BTC must finish above {order.target_btc_price:.2f}"
            if order.side == "UP"
            else f"BTC must finish below {order.target_btc_price:.2f}"
        )
        print(f"  order_{idx}_market_slug    = {order.market_slug}")
        print(f"  order_{idx}_market_title   = {order.market_title}")
        print(f"  order_{idx}_side           = {order.side}")
        print(f"  order_{idx}_shares         = {_fmt(order.shares)}")
        print(f"  order_{idx}_entry_price    = {_fmt(order.entry_price)}")
        print(f"  order_{idx}_entry_btc      = {order.entry_btc_price:.2f}")
        print(f"  order_{idx}_period_open_price_to_beat = {order.target_btc_price:.2f}")
        print(f"  order_{idx}_win_condition  = {win_condition}")
        print(f"  order_{idx}_target         = {describe_target(order)}")
        print(f"  order_{idx}_current_btc    = {current_btc_price:.2f}")
        print(f"  order_{idx}_position_state = {status}")


def print_features(features, debug: bool) -> None:
    print("Features:")
    print(f"  btc_price             = {features.price_usd:.2f}")
    print(f"  delta_prev_tick       = {features.delta_from_previous_tick}")
    print(f"  momentum_1m           = {features.momentum_1m}")
    print(f"  momentum_5m           = {features.momentum_5m}")
    print(f"  velocity_15s          = {features.velocity_15s}")
    print(f"  velocity_30s          = {features.velocity_30s}")
    print(f"  momentum_acceleration = {features.momentum_acceleration}")
    print(f"  volatility_5m         = {features.volatility_5m}")
    print(f"  consecutive_flat_ticks= {features.consecutive_flat_ticks}")
    print(f"  directional_ticks     = {features.consecutive_directional_ticks}")
    print(f"  rsi_9                 = {features.rsi_9}")
    print(f"  rsi_14                = {features.rsi_14}")
    print(f"  rsi_speed_divergence  = {features.rsi_speed_divergence}")
    print(f"  ema_9                 = {features.ema_9}")
    print(f"  ema_21                = {features.ema_21}")
    print(f"  ema_alignment         = {features.ema_alignment}")
    print(f"  ema_cross_direction   = {features.ema_cross_direction}")
    print(f"  adx_14                = {features.adx_14}")
    print(f"  atr_14                = {features.atr_14}")
    print(f"  last_10_ticks_dir     = {features.last_10_ticks_direction}")
    if not debug:
        return

    print(f"  window_open_price     = {features.window_open_price:.2f}")
    print(f"  delta_from_window_pct = {features.delta_pct_from_window_open * 100:.4f}%")
    print(f"  trailing_5m_open_price= {features.trailing_5m_open_price:.2f}")
    print(f"  delta_from_5m_pct     = {features.delta_pct_from_trailing_5m_open * 100:.4f}%")
    print(f"  retained_samples      = {features.retained_sample_count}")
    print(f"  window_samples        = {features.window_sample_count}")
    print(f"  trailing_5m_samples   = {features.trailing_5m_sample_count}")


def print_market_context(market, debug: bool) -> None:
    print("Market:")
    print(f"  slug                  = {market.slug}")
    print(f"  period_open_price_to_beat = {_fmt(market.settlement_threshold)}")
    print(f"  up_market_probability = {_fmt(getattr(market, 'up_market_probability', None))}")
    print(f"  down_market_probability = {_fmt(getattr(market, 'down_market_probability', None))}")
    if not debug:
        return

    print(f"  title                 = {market.title}")
    print(f"  question              = {market.question}")
    print(f"  event_id              = {market.event_id}")
    print(f"  market_id             = {market.market_id}")
    print(f"  up_token              = {market.up_token_id}")
    print(f"  down_token            = {market.down_token_id}")


def print_llm_skip_reason(reason: str) -> None:
    print("LLM decision skipped:")
    print(f"  reason            = {reason}")


def both_sides_untradable_reason(up_snapshot: TokenQuoteSnapshot, down_snapshot: TokenQuoteSnapshot) -> str:
    return (
        "Both sides are currently not safe to submit. "
        f"UP: {up_snapshot.submit_reason} | "
        f"DOWN: {down_snapshot.submit_reason}"
    )


def print_llm_decision(decision, market, features, debug: bool) -> None:
    print("LLM decision:")
    print(f"  side              = {decision.side}")
    print(f"  confidence        = {decision.confidence:.3f}")
    print(f"  effective_conf    = {_effective_confidence(decision, market=market, features=features):.3f}")
    print(f"  max_price_to_pay  = {decision.max_price_to_pay:.3f}")
    print(f"  reason            = {decision.reason}")
    if debug and getattr(decision, "prompt_text", None):
        print("LLM prompt:")
        print(decision.prompt_text)


def print_trade_execution_result(result, debug: bool) -> None:
    print("Trade execution result:")
    print(f"  executed = {result.executed}")
    print(f"  side     = {result.side}")
    print(f"  size     = {result.size:.4f}")
    print(f"  price    = {_fmt(result.price)}")
    print(f"  token_id = {result.token_id}")
    print(f"  reason   = {result.reason}")


def run_once() -> None:
    global _FIRST_LOOP
    global _SESSION_LOSS_TRADES
    cfg = get_trading_config()
    use_recommended_limit = getattr(cfg, "use_recommended_limit", True)
    enforce_session_loss_trade_limit(cfg)
    if cfg.debug:
        print(f"[{datetime.now(timezone.utc).isoformat()}] BTC up/down agent tick")

    market = find_current_btc_updown_market()
    if not market:
        if _FIRST_LOOP:
            print_account_snapshot(debug=cfg.debug)
            _FIRST_LOOP = False
        if cfg.debug:
            print("No BTC Up/Down market found.")
        return

    previous_state = get_state()
    previous_orders = list(getattr(previous_state, "active_orders", []))
    previous_market_slug = getattr(previous_state, "market_slug", None)
    period_changed = sync_period_state(market.slug, market.title)
    state = get_state()
    if _FIRST_LOOP or period_changed:
        account = get_account_balance_snapshot()
        print_account_snapshot_from_snapshot(account, debug=cfg.debug)
        enforce_minimum_wallet_balance(account)
        _FIRST_LOOP = False

    market = resolve_price_to_beat_with_retries(market)
    if period_changed:
        final_resolution_btc_price = None
        try:
            if has_valid_price_to_beat(market.settlement_threshold):
                final_resolution_btc_price = float(market.settlement_threshold)
            elif previous_market_slug:
                final_resolution_btc_price = fetch_btc_resolution_price_for_slug(previous_market_slug)
        except Exception:
            final_resolution_btc_price = None
        if previous_orders:
            try:
                _SESSION_LOSS_TRADES += finalize_completed_orders(
                    previous_orders,
                    final_resolution_btc_price if final_resolution_btc_price is not None else fetch_btc_spot_price(),
                )
            except Exception:
                pass
        if previous_market_slug:
            finalize_pending_period_log(previous_market_slug, final_resolution_btc_price)
        print(f"New 5-minute market period detected: {market.slug}")
        clear_price_to_beat_debug_files()
        _DEBUG_WRITTEN_SLUGS.clear()
        enforce_session_loss_trade_limit(cfg)
    if cfg.debug:
        write_price_to_beat_debug_file(market.slug)
    if not has_valid_price_to_beat(market.settlement_threshold):
        print_market_context(market, debug=cfg.debug)
        if not cfg.debug:
            write_price_to_beat_debug_file(market.slug, force=True)
        print(
            "ERROR: Invalid period_open_price_to_beat for current market. "
            "Aborting BTC agent execution."
        )
        sys.exit(1)

    if state.trades_executed >= cfg.max_trades_per_period:
        if cfg.debug:
            print(
                f"Trade limit reached for current period: "
                f"{state.trades_executed}/{cfg.max_trades_per_period}"
            )
        active_orders = get_active_orders()
        if active_orders:
            up_snapshot = None
            down_snapshot = None
            up_snapshot = get_token_quote_snapshot(market.up_token_id)
            down_snapshot = get_token_quote_snapshot(market.down_token_id)
            try:
                features = build_btc_features(window_start_ts=market.start_ts)
                current_btc_price = features.price_usd
                observed_at = features.as_of
            except Exception:
                features = None
                current_btc_price = fetch_btc_spot_price()
                observed_at = None
            update_active_order_logs(
                current_btc_price,
                observed_at=observed_at,
                features=features,
                up_snapshot=up_snapshot,
                down_snapshot=down_snapshot,
            )
        else:
            current_btc_price = fetch_btc_spot_price()
        print_active_orders(current_btc_price)
        if cfg.debug:
            print("-" * 80)
        return

    if cfg.max_trades_per_period > 1 and state.trades_executed > 0:
        cooldown_loops_remaining = get_trade_cooldown_remaining()
        if cooldown_loops_remaining > 0:
            print_market_context(market, debug=cfg.debug)
            try:
                features = build_btc_features(window_start_ts=market.start_ts)
                print_features(features, debug=cfg.debug)
                current_btc_price = features.price_usd
                observed_at = features.as_of
            except Exception:
                features = None
                current_btc_price = fetch_btc_spot_price()
                observed_at = None

            up_snapshot = None
            down_snapshot = None
            active_orders = get_active_orders()
            if active_orders:
                up_snapshot = get_token_quote_snapshot(market.up_token_id)
                down_snapshot = get_token_quote_snapshot(market.down_token_id)

            if active_orders:
                update_active_order_logs(
                    current_btc_price,
                    observed_at=observed_at,
                    features=features,
                    up_snapshot=up_snapshot,
                    down_snapshot=down_snapshot,
                )
                print_active_orders(current_btc_price)

            consume_trade_cooldown_loop()
            print_llm_skip_reason(
                "trade cooldown active after prior execution "
                f"({cooldown_loops_remaining}/3 loops remaining before another trade is allowed)"
            )
            if cfg.debug:
                print("-" * 80)
            return

    print_market_context(market, debug=cfg.debug)

    up_snapshot = None
    down_snapshot = None
    if use_recommended_limit:
        up_snapshot = get_token_quote_snapshot(market.up_token_id)
        down_snapshot = get_token_quote_snapshot(market.down_token_id)
        print_quote_snapshot_from_snapshot("UP", up_snapshot, debug=cfg.debug)
        print_quote_snapshot_from_snapshot("DOWN", down_snapshot, debug=cfg.debug)

        if not up_snapshot.ok_to_submit and not down_snapshot.ok_to_submit:
            skip_reason = both_sides_untradable_reason(up_snapshot, down_snapshot)
            append_pending_period_tick_analysis(
                market,
                up_snapshot=up_snapshot,
                down_snapshot=down_snapshot,
                skip_reason=skip_reason,
            )
            print_llm_skip_reason(skip_reason)
            if cfg.debug:
                print("-" * 80)
            return

    features = build_btc_features(window_start_ts=market.start_ts)
    print_features(features, debug=cfg.debug)
    features_ready, feature_skip_reason = get_feature_readiness(features)
    if not features_ready:
        append_pending_period_tick_analysis(
            market,
            up_snapshot=up_snapshot,
            down_snapshot=down_snapshot,
            features=features,
            skip_reason=feature_skip_reason,
            observed_at=features.as_of,
        )
        print_llm_skip_reason(feature_skip_reason)
        if cfg.debug:
            print("-" * 80)
        return

    if up_snapshot is None:
        up_snapshot = get_token_quote_snapshot(market.up_token_id)
    if down_snapshot is None:
        down_snapshot = get_token_quote_snapshot(market.down_token_id)

    if cfg.max_trades_per_period > 1 and state.trades_executed > 0:
        losing_active_orders = _get_losing_active_orders(features.price_usd)
        if losing_active_orders:
            skip_reason = (
                "existing active order is currently losing; "
                "skipping additional same-period trade evaluation"
            )
            append_pending_period_tick_analysis(
                market,
                up_snapshot=up_snapshot,
                down_snapshot=down_snapshot,
                features=features,
                skip_reason=skip_reason,
                observed_at=features.as_of,
            )
            update_active_order_logs(
                features.price_usd,
                observed_at=features.as_of,
                features=features,
                up_snapshot=up_snapshot,
                down_snapshot=down_snapshot,
            )
            print_active_orders(features.price_usd)
            print_llm_skip_reason(skip_reason)
            if cfg.debug:
                print("-" * 80)
            return

    decision = decide_trade(features, market, up_snapshot=up_snapshot, down_snapshot=down_snapshot)
    append_pending_period_tick_analysis(
        market,
        up_snapshot=up_snapshot,
        down_snapshot=down_snapshot,
        features=features,
        decision=decision,
        observed_at=features.as_of,
    )
    print_llm_decision(decision, market=market, features=features, debug=cfg.debug)

    decision_snapshot = None
    if decision.side == "UP":
        if use_recommended_limit:
            decision_snapshot = get_decision_quote_snapshot(
                market,
                decision,
                up_snapshot,
                down_snapshot,
            )
            if cfg.debug:
                print_quote_snapshot_from_snapshot("UP (with decision)", decision_snapshot, debug=True)
        else:
            decision_snapshot = get_token_quote_snapshot(market.up_token_id, decision=decision)
    elif decision.side == "DOWN":
        if use_recommended_limit:
            decision_snapshot = get_decision_quote_snapshot(
                market,
                decision,
                up_snapshot,
                down_snapshot,
            )
            if cfg.debug:
                print_quote_snapshot_from_snapshot("DOWN (with decision)", decision_snapshot, debug=True)
        else:
            decision_snapshot = get_token_quote_snapshot(market.down_token_id, decision=decision)

    try:
        result = maybe_execute_trade(market, decision, features=features, snapshot=decision_snapshot)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    if result.execution_snapshot is not None and cfg.debug:
        print_quote_snapshot_from_snapshot("Execution", result.execution_snapshot, debug=True)
    print_trade_execution_result(result, debug=cfg.debug)

    if (
        not result.executed
        and decision.side in ("UP", "DOWN")
        and result.token_id
    ):
        append_failed_order_attempt(
            market,
            decision,
            result,
            features=features,
            up_snapshot=up_snapshot,
            down_snapshot=down_snapshot,
            observed_at=features.as_of,
            trade_number_in_period=state.trades_executed + 1,
        )

    if result.executed:
        if cfg.max_trades_per_period > 1:
            set_trade_cooldown(3)

    if result.executed and decision.side in ("UP", "DOWN") and result.token_id:
        target_btc_price = market.settlement_threshold
        target_is_approximate = target_btc_price is None
        if target_btc_price is None:
            target_btc_price = (
                estimate_market_window_reference_price(
                    market.start_ts,
                    now=features.as_of,
                )
                or features.window_open_price
            )

        new_order = ActivePaperOrder(
            market_slug=market.slug,
            market_title=market.title,
            trade_number_in_period=state.trades_executed + 1,
            side=decision.side,
            shares=result.size,
            entry_price=result.price,
            token_id=result.token_id,
            target_btc_price=target_btc_price,
            entry_btc_price=features.price_usd,
            target_is_approximate=target_is_approximate,
        )
        promote_pending_period_log_to_completed(
            market.slug,
            trade_number_in_period=new_order.trade_number_in_period,
        )
        record_executed_trade(new_order)
        append_completed_order_tick(
            new_order,
            current_btc_price=features.price_usd,
            phase="PLACED",
            observed_at=features.as_of,
            features=features,
            up_snapshot=up_snapshot,
            down_snapshot=down_snapshot,
        )

    enforce_session_loss_trade_limit(cfg)

    active_btc_price = features.price_usd
    update_active_order_logs(
        active_btc_price,
        observed_at=features.as_of,
        features=features,
        up_snapshot=up_snapshot,
        down_snapshot=down_snapshot,
    )
    if cfg.debug and get_active_orders():
        print_active_orders(active_btc_price)
    if cfg.debug:
        print("-" * 80)


def enforce_allowed_ip_location() -> None:
    cfg = get_trading_config()
    if cfg.llm_connection_debug:
        print("Skipping public IP geolocation check because LLM_CONNECTION_DEBUG=true")
        return

    print("Checking public IP geolocation...")
    public_ip, location, ip_is_allowed = check_current_public_ip_location()
    print_ip_location(public_ip, location, debug=cfg.debug)

    if not ip_is_allowed:
        print(
            "ERROR: Public IP geolocation is not in an allowed country "
            "(Indonesia or Mexico). Aborting BTC agent startup."
        )
        sys.exit(1)


def main() -> None:
    cfg = get_trading_config()
    print(f"Network proxy: {describe_proxy_configuration()}")
    if cfg.debug:
        print(f"Starting BTC agent (paper_trading={cfg.paper_trading})")

    if cfg.llm_connection_debug:
        print("LLM connection debug mode enabled.")
        success, detail = test_llm_connection()
        if success:
            print(f"LLM connection test: {detail}")
            return
        print(f"ERROR: LLM connection test failed: {detail}")
        sys.exit(1)

    enforce_allowed_ip_location()

    startup_account = get_account_balance_snapshot()
    enforce_minimum_wallet_balance(startup_account)

    interval = int(os.getenv("BTC_AGENT_LOOP_INTERVAL", "30"))
    print("Press q to quit.")

    monitor_context = QuitKeyMonitor()
    try:
        with monitor_context as quit_monitor:
            while True:
                if quit_monitor.poll_quit_requested():
                    finalize_current_period_logs_on_exit()
                    print("Quit requested via keyboard. Exiting BTC agent.")
                    return
                run_once()
                print(f"Sleeping {interval} seconds before next tick...")
                if wait_for_next_tick_or_quit(interval, quit_monitor=quit_monitor):
                    finalize_current_period_logs_on_exit()
                    print("Quit requested via keyboard. Exiting BTC agent.")
                    return
    except KeyboardInterrupt:
        finalize_current_period_logs_on_exit()
        print("Keyboard interrupt received. Exiting BTC agent.")
        return


if __name__ == "__main__":
    main()
