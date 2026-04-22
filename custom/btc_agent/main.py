# custom/btc_agent/main.py

from datetime import datetime, timezone
import os
import time
import sys

from .config import get_trading_config
from .market_lookup import find_current_btc_updown_market
from .indicators import build_btc_features, fetch_btc_spot_price
from .llm_decision import decide_trade
from .executor import (
    TokenQuoteSnapshot,
    get_account_balance_snapshot,
    get_token_quote_snapshot,
    maybe_execute_trade,
)
from .paper_state import (
    ActivePaperOrder,
    classify_position,
    describe_target,
    get_active_orders,
    get_state,
    record_executed_trade,
    sync_period_state,
)
from scripts.python.check_public_ip_indonesia import (
    check_current_public_ip_location,
    is_allowed_location,
)


_FIRST_LOOP = True


def _fmt(value):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_ip_location(public_ip, location, debug: bool) -> None:
    print(f"public_ip: {public_ip or 'unknown'}")
    print(f"is_allowed_location: {str(is_allowed_location(location)).lower()}")
    print(f"lookup_success: {str(bool(location.get('success', False))).lower()}")
    print(f"country: {location.get('country', 'unknown')}")
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

    message = location.get("message")
    if message:
        print(f"message: {message}")


def print_quote_snapshot(label: str, token_id: str, decision=None, debug: bool = True) -> None:
    q = get_token_quote_snapshot(token_id, decision=decision)
    print_quote_snapshot_from_snapshot(label, q, debug=debug)


def print_quote_snapshot_from_snapshot(
    label: str,
    q: TokenQuoteSnapshot,
    debug: bool = True,
) -> None:
    print(f"{label} quote snapshot:")
    if not debug:
        print(f"  buy_quote              = {_fmt(q.buy_quote)}")
        print(f"  recommended_limit_price= {_fmt(q.recommended_limit_price)}")
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


def get_decision_quote_snapshot(market, decision) -> TokenQuoteSnapshot:
    token_id = market.up_token_id if decision.side == "UP" else market.down_token_id
    return get_token_quote_snapshot(token_id, decision=decision)


def print_account_snapshot(debug: bool) -> None:
    account = get_account_balance_snapshot()
    print("Account balances:")
    if not debug:
        print(f"  cash_balance_usdc      = {_fmt(account.cash_balance)}")
        print(f"  portfolio_balance_usd  = {_fmt(account.portfolio_balance)}")
        print(f"  total_account_value_usd= {_fmt(account.total_account_value)}")
        return

    print(f"  signer_address         = {account.signer_address}")
    print(f"  balance_address        = {account.balance_address}")
    print(f"  proxy_address          = {account.proxy_address or 'None'}")
    print(f"  cash_balance_usdc      = {_fmt(account.cash_balance)}")
    print(f"  portfolio_balance_usd  = {_fmt(account.portfolio_balance)}")
    print(f"  total_account_value_usd= {_fmt(account.total_account_value)}")
    print(f"  balance_error          = {account.error or 'None'}")


def print_active_orders(current_btc_price: float) -> None:
    active_orders = get_active_orders()
    if not active_orders:
        print("Active orders: None")
        return

    print("Active orders:")
    for idx, order in enumerate(active_orders, start=1):
        status = classify_position(order, current_btc_price)
        print(f"  order_{idx}_market_slug    = {order.market_slug}")
        print(f"  order_{idx}_market_title   = {order.market_title}")
        print(f"  order_{idx}_side           = {order.side}")
        print(f"  order_{idx}_shares         = {_fmt(order.shares)}")
        print(f"  order_{idx}_entry_price    = {_fmt(order.entry_price)}")
        print(f"  order_{idx}_entry_btc      = {order.entry_btc_price:.2f}")
        print(f"  order_{idx}_target         = {describe_target(order)}")
        print(f"  order_{idx}_current_btc    = {current_btc_price:.2f}")
        print(f"  order_{idx}_position_state = {status}")


def print_features(features, debug: bool) -> None:
    print("Features:")
    print(f"  btc_price             = {features.price_usd:.2f}")
    print(f"  momentum_5m           = {features.momentum_5m}")
    print(f"  volatility_5m         = {features.volatility_5m}")
    if not debug:
        return

    print(f"  window_open_price     = {features.window_open_price:.2f}")
    print(f"  delta_from_window_pct = {features.delta_pct_from_window_open * 100:.4f}%")
    print(f"  rsi_14                = {features.rsi_14}")


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
    cfg = get_trading_config()
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

    period_changed = sync_period_state(market.slug, market.title)
    state = get_state()
    if _FIRST_LOOP or period_changed:
        print_account_snapshot(debug=cfg.debug)
        _FIRST_LOOP = False
    if period_changed:
        print(f"New 5-minute market period detected: {market.slug}")

    if state.trades_executed >= cfg.max_trades_per_period:
        if cfg.debug:
            print(
                f"Trade limit reached for current period: "
                f"{state.trades_executed}/{cfg.max_trades_per_period}"
            )
        print_active_orders(fetch_btc_spot_price())
        if cfg.debug:
            print("-" * 80)
        return

    if cfg.debug:
        print("Market found:")
        print(f"  title      = {market.title}")
        print(f"  question   = {market.question}")
        print(f"  slug       = {market.slug}")
        print(f"  event_id   = {market.event_id}")
        print(f"  market_id  = {market.market_id}")
        print(f"  up_token   = {market.up_token_id}")
        print(f"  down_token = {market.down_token_id}")
        print(f"  threshold  = {_fmt(market.settlement_threshold)}")

    print_quote_snapshot("UP", market.up_token_id, debug=cfg.debug)
    print_quote_snapshot("DOWN", market.down_token_id, debug=cfg.debug)

    features = build_btc_features(window_start_ts=market.start_ts)
    print_features(features, debug=cfg.debug)

    decision = decide_trade(features, market)
    print_llm_decision(decision, debug=cfg.debug)

    decision_snapshot = None
    if decision.side == "UP":
        decision_snapshot = get_decision_quote_snapshot(market, decision)
        if cfg.debug:
            print_quote_snapshot_from_snapshot("UP (with decision)", decision_snapshot, debug=True)
    elif decision.side == "DOWN":
        decision_snapshot = get_decision_quote_snapshot(market, decision)
        if cfg.debug:
            print_quote_snapshot_from_snapshot("DOWN (with decision)", decision_snapshot, debug=True)

    try:
        result = maybe_execute_trade(market, decision, snapshot=decision_snapshot)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    if result.execution_snapshot is not None and cfg.debug:
        print_quote_snapshot_from_snapshot("Execution", result.execution_snapshot, debug=True)
    print_trade_execution_result(result, debug=cfg.debug)

    if result.executed and decision.side in ("UP", "DOWN") and result.token_id:
        target_btc_price = market.settlement_threshold
        target_is_approximate = target_btc_price is None
        if target_btc_price is None:
            target_btc_price = features.window_open_price

        record_executed_trade(
            ActivePaperOrder(
                market_slug=market.slug,
                market_title=market.title,
                side=decision.side,
                shares=result.size,
                entry_price=result.price,
                token_id=result.token_id,
                target_btc_price=target_btc_price,
                entry_btc_price=features.price_usd,
                target_is_approximate=target_is_approximate,
            )
        )

    active_btc_price = features.price_usd
    if cfg.debug and get_active_orders():
        print_active_orders(active_btc_price)
    if cfg.debug:
        print("-" * 80)


def enforce_allowed_ip_location() -> None:
    cfg = get_trading_config()
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
    if cfg.debug:
        print(f"Starting BTC agent (paper_trading={cfg.paper_trading})")
    enforce_allowed_ip_location()

    interval = int(os.getenv("BTC_AGENT_LOOP_INTERVAL", "30"))

    while True:
        run_once()
        print(f"Sleeping {interval} seconds before next tick...")
        time.sleep(interval)


if __name__ == "__main__":
    main()
