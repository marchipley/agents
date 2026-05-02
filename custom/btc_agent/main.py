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
_SESSION_AUTOMATED_TRADES = 0
_SESSION_FIRST_TRADE_WALLET_VALUE = None
_SESSION_PENDING_EXIT_AFTER_PERIOD = False


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


def _completed_order_final_log_path(
    market_slug: str,
    outcome_label: str,
    trade_number_in_period: Optional[int] = None,
) -> str:
    completed_orders_dir = os.path.join(os.getcwd(), "completed_orders")
    os.makedirs(completed_orders_dir, exist_ok=True)
    return os.path.join(
        completed_orders_dir,
        f"completed_order_{outcome_label}_{_extract_slug_timestamp(market_slug)}{_trade_number_suffix(trade_number_in_period)}.txt",
    )


def _classify_outcome_label(position_state: str) -> str:
    if position_state == "WINNING":
        return "win"
    if position_state == "LOSING":
        return "loss"
    return "tied"


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
        f"{prefix}_spread={_fmt(getattr(snapshot, 'spread', None))}",
    ]


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
                    f"volatility_5m={getattr(features, 'volatility_5m', None)}",
                    f"window_open_price={_fmt(getattr(features, 'window_open_price', None))}",
                    f"delta_from_window_pct={((getattr(features, 'delta_pct_from_window_open', 0.0) or 0.0) * 100):.4f}%",
                    f"trailing_5m_open_price={_fmt(getattr(features, 'trailing_5m_open_price', None))}",
                    f"delta_from_5m_pct={((getattr(features, 'delta_pct_from_trailing_5m_open', 0.0) or 0.0) * 100):.4f}%",
                    f"rsi_14={getattr(features, 'rsi_14', None)}",
                ]
            )

        if decision is not None:
            lines.extend(
                [
                    f"decision_side={decision.side}",
                    f"decision_confidence={decision.confidence:.3f}",
                    f"decision_max_price_to_pay={decision.max_price_to_pay:.3f}",
                    f"decision_reason={decision.reason}",
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
        try:
            os.remove(pending_path)
        except FileNotFoundError:
            pass
        return
    os.replace(pending_path, completed_path)


def delete_pending_period_log(market_slug: str) -> None:
    if not market_slug:
        return
    try:
        os.remove(_pending_period_log_path(market_slug))
    except FileNotFoundError:
        pass


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
                        f"feature_volatility_5m={getattr(features, 'volatility_5m', None)}",
                        f"feature_window_open_price={_fmt(getattr(features, 'window_open_price', None))}",
                        f"feature_delta_from_window_pct={((getattr(features, 'delta_pct_from_window_open', 0.0) or 0.0) * 100):.4f}%",
                        f"feature_trailing_5m_open_price={_fmt(getattr(features, 'trailing_5m_open_price', None))}",
                        f"feature_delta_from_5m_pct={((getattr(features, 'delta_pct_from_trailing_5m_open', 0.0) or 0.0) * 100):.4f}%",
                        f"feature_rsi_14={getattr(features, 'rsi_14', None)}",
                        "",
                    ]
                )
            )
        if up_snapshot is not None:
            log_file.write("\n".join(_snapshot_summary("active_up", up_snapshot) + [""]))
        if down_snapshot is not None:
            log_file.write("\n".join(_snapshot_summary("active_down", down_snapshot) + [""]))
    if phase == "COMPLETED":
        final_path = _completed_order_final_log_path(
            order.market_slug,
            outcome_label,
            getattr(order, "trade_number_in_period", None),
        )
        try:
            os.replace(log_path, final_path)
        except FileNotFoundError:
            pass


def finalize_completed_orders(previous_orders, current_btc_price: float) -> None:
    for order in previous_orders:
        append_completed_order_tick(
            order,
            current_btc_price=current_btc_price,
            phase="COMPLETED",
        )


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


def _get_wallet_value_for_limit_check(account: AccountBalanceSnapshot):
    if account is None:
        return None
    total_account_value = getattr(account, "total_account_value", None)
    cash_balance = getattr(account, "cash_balance", None)
    if total_account_value is not None:
        return float(total_account_value)
    if cash_balance is not None:
        return float(cash_balance)
    return None


def enforce_session_trade_limit(cfg) -> None:
    global _SESSION_PENDING_EXIT_AFTER_PERIOD
    if getattr(cfg, "max_automated_trades", 0) <= 0:
        return
    if _SESSION_AUTOMATED_TRADES < cfg.max_automated_trades:
        return
    if _SESSION_FIRST_TRADE_WALLET_VALUE is None:
        return
    current_account = get_account_balance_snapshot()
    current_wallet_value = _get_wallet_value_for_limit_check(current_account)
    if current_wallet_value is None:
        return
    if current_wallet_value >= _SESSION_FIRST_TRADE_WALLET_VALUE:
        return
    if get_active_orders():
        _SESSION_PENDING_EXIT_AFTER_PERIOD = True
        return
    print(
        "Max automated trades for this session has been reached with a net loss "
        f"({_SESSION_AUTOMATED_TRADES}/{cfg.max_automated_trades}; "
        f"first_trade_wallet_value={_SESSION_FIRST_TRADE_WALLET_VALUE:.3f}; "
        f"current_wallet_value={current_wallet_value:.3f}). "
        "Exiting BTC agent."
    )
    sys.exit(0)


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
    print(f"  volatility_5m         = {features.volatility_5m}")
    if not debug:
        return

    print(f"  window_open_price     = {features.window_open_price:.2f}")
    print(f"  delta_from_window_pct = {features.delta_pct_from_window_open * 100:.4f}%")
    print(f"  trailing_5m_open_price= {features.trailing_5m_open_price:.2f}")
    print(f"  delta_from_5m_pct     = {features.delta_pct_from_trailing_5m_open * 100:.4f}%")
    print(f"  rsi_14                = {features.rsi_14}")
    print(f"  retained_samples      = {features.retained_sample_count}")
    print(f"  window_samples        = {features.window_sample_count}")
    print(f"  trailing_5m_samples   = {features.trailing_5m_sample_count}")


def print_market_context(market, debug: bool) -> None:
    print("Market:")
    print(f"  slug                  = {market.slug}")
    print(f"  period_open_price_to_beat = {_fmt(market.settlement_threshold)}")
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


def print_llm_decision(decision, debug: bool) -> None:
    print("LLM decision:")
    print(f"  side              = {decision.side}")
    print(f"  confidence        = {decision.confidence:.3f}")
    print(f"  max_price_to_pay  = {decision.max_price_to_pay:.3f}")
    print(f"  reason            = {decision.reason}")


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
    global _SESSION_AUTOMATED_TRADES
    global _SESSION_FIRST_TRADE_WALLET_VALUE
    global _SESSION_PENDING_EXIT_AFTER_PERIOD
    cfg = get_trading_config()
    use_recommended_limit = getattr(cfg, "use_recommended_limit", True)
    enforce_session_trade_limit(cfg)
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
        if previous_orders:
            try:
                if has_valid_price_to_beat(market.settlement_threshold):
                    final_resolution_btc_price = float(market.settlement_threshold)
                elif previous_market_slug:
                    final_resolution_btc_price = fetch_btc_resolution_price_for_slug(previous_market_slug)
                finalize_completed_orders(
                    previous_orders,
                    final_resolution_btc_price if final_resolution_btc_price is not None else fetch_btc_spot_price(),
                )
            except Exception:
                pass
        elif previous_market_slug:
            delete_pending_period_log(previous_market_slug)
        print(f"New 5-minute market period detected: {market.slug}")
        clear_price_to_beat_debug_files()
        _DEBUG_WRITTEN_SLUGS.clear()
        if _SESSION_PENDING_EXIT_AFTER_PERIOD:
            _SESSION_PENDING_EXIT_AFTER_PERIOD = False
            enforce_session_trade_limit(cfg)
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
            if use_recommended_limit:
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
            if active_orders and use_recommended_limit:
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

    decision = decide_trade(features, market, up_snapshot=up_snapshot, down_snapshot=down_snapshot)
    append_pending_period_tick_analysis(
        market,
        up_snapshot=up_snapshot,
        down_snapshot=down_snapshot,
        features=features,
        decision=decision,
        observed_at=features.as_of,
    )
    print_llm_decision(decision, debug=cfg.debug)

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

    first_trade_wallet_baseline = _SESSION_FIRST_TRADE_WALLET_VALUE
    if first_trade_wallet_baseline is None:
        baseline_account = get_account_balance_snapshot()
        first_trade_wallet_baseline = _get_wallet_value_for_limit_check(baseline_account)

    try:
        result = maybe_execute_trade(market, decision, features=features, snapshot=decision_snapshot)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    if result.execution_snapshot is not None and cfg.debug:
        print_quote_snapshot_from_snapshot("Execution", result.execution_snapshot, debug=True)
    print_trade_execution_result(result, debug=cfg.debug)

    if result.executed:
        if _SESSION_FIRST_TRADE_WALLET_VALUE is None:
            _SESSION_FIRST_TRADE_WALLET_VALUE = first_trade_wallet_baseline
        _SESSION_AUTOMATED_TRADES += 1
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

    enforce_session_trade_limit(cfg)

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
                    print("Quit requested via keyboard. Exiting BTC agent.")
                    return
                run_once()
                print(f"Sleeping {interval} seconds before next tick...")
                if wait_for_next_tick_or_quit(interval, quit_monitor=quit_monitor):
                    print("Quit requested via keyboard. Exiting BTC agent.")
                    return
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting BTC agent.")
        return


if __name__ == "__main__":
    main()
