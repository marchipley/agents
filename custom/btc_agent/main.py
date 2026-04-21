# custom/btc_agent/main.py

from datetime import datetime, timezone
import os
import time

from .config import get_trading_config
from .market_lookup import find_current_btc_updown_market
from .indicators import build_btc_features
from .llm_decision import decide_trade
from .executor import (
    get_account_balance_snapshot,
    get_token_quote_snapshot,
    maybe_execute_paper_trade,
)


def _fmt(value):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_quote_snapshot(label: str, token_id: str, decision=None) -> None:
    q = get_token_quote_snapshot(token_id, decision=decision)
    print(f"{label} quote snapshot:")
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


def print_account_snapshot() -> None:
    account = get_account_balance_snapshot()
    print("Account balances:")
    print(f"  signer_address         = {account.signer_address}")
    print(f"  balance_address        = {account.balance_address}")
    print(f"  proxy_address          = {account.proxy_address or 'None'}")
    print(f"  cash_balance_usdc      = {_fmt(account.cash_balance)}")
    print(f"  portfolio_balance_usd  = {_fmt(account.portfolio_balance)}")
    print(f"  total_account_value_usd= {_fmt(account.total_account_value)}")
    print(f"  balance_error          = {account.error or 'None'}")

def run_once() -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] BTC up/down agent tick")
    print_account_snapshot()

    market = find_current_btc_updown_market()
    if not market:
        print("No BTC Up/Down market found.")
        return

    print("Market found:")
    print(f"  title      = {market.title}")
    print(f"  slug       = {market.slug}")
    print(f"  event_id   = {market.event_id}")
    print(f"  market_id  = {market.market_id}")
    print(f"  up_token   = {market.up_token_id}")
    print(f"  down_token = {market.down_token_id}")

    print_quote_snapshot("UP", market.up_token_id)
    print_quote_snapshot("DOWN", market.down_token_id)

    features = build_btc_features(window_start_ts=market.start_ts)
    print("Features:")
    print(f"  btc_price             = {features.price_usd:.2f}")
    print(f"  window_open_price     = {features.window_open_price:.2f}")
    print(f"  delta_from_window_pct = {features.delta_pct_from_window_open * 100:.4f}%")
    print(f"  rsi_14                = {features.rsi_14}")
    print(f"  momentum_5m           = {features.momentum_5m}")
    print(f"  volatility_5m         = {features.volatility_5m}")

    decision = decide_trade(features, market)
    print("LLM decision:")
    print(f"  side              = {decision.side}")
    print(f"  confidence        = {decision.confidence:.3f}")
    print(f"  max_price_to_pay  = {decision.max_price_to_pay:.3f}")
    print(f"  reason            = {decision.reason}")

    if decision.side == "UP":
        print_quote_snapshot("UP (with decision)", market.up_token_id, decision=decision)
    elif decision.side == "DOWN":
        print_quote_snapshot("DOWN (with decision)", market.down_token_id, decision=decision)

    result = maybe_execute_paper_trade(market, decision)
    print("Paper execution result:")
    print(f"  executed = {result.executed}")
    print(f"  side     = {result.side}")
    print(f"  size     = {result.size:.4f}")
    print(f"  price    = {_fmt(result.price)}")
    print(f"  token_id = {result.token_id}")
    print(f"  reason   = {result.reason}")
    print("-" * 80)


def main() -> None:
    cfg = get_trading_config()
    print(f"Starting BTC agent (paper_trading={cfg.paper_trading})")

    interval = int(os.getenv("BTC_AGENT_LOOP_INTERVAL", "30"))

    while True:
        run_once()
        print(f"Sleeping {interval} seconds before next tick...")
        time.sleep(interval)


if __name__ == "__main__":
    main()
