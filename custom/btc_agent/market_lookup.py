# custom/btc_agent/market_lookup.py

import json
import os
import requests
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .config import get_polymarket_config, get_trading_config

@dataclass
class BtcUpDownMarket:
    event_id: str
    market_id: str
    up_token_id: str
    down_token_id: str
    title: str
    slug: str
    start_ts: int
    end_ts: int

def _current_btc_5m_slug() -> str:
    """
    Polymarket BTC 5-minute window slug is:
      btc-updown-5m-<window_start_ts>
    where window_start_ts is a Unix timestamp aligned to 5-minute boundaries.
    """
    now_ts = int(datetime.now(timezone.utc).timestamp())
    window_start = now_ts - (now_ts % 300)
    return f"btc-updown-5m-{window_start}"

def _fetch_event_by_slug(slug: str) -> dict:
    cfg = get_polymarket_config()
    url = f"{cfg.gamma_api}/events/slug/{slug}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def _parse_clob_token_ids(value):
    if not value:
        return None, None

    if isinstance(value, list) and len(value) >= 2:
        return str(value[0]), str(value[1])

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list) and len(parsed) >= 2:
                return str(parsed[0]), str(parsed[1])
        except Exception:
            parts = [p.strip() for p in value.split(",") if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[1]

    return None, None

def _extract_market_from_event(event: dict, slug: str) -> Optional[BtcUpDownMarket]:
    title = event.get("title", "")
    markets = event.get("markets") or []
    if not markets:
        return None

    m = markets[0]

    up_token = None
    down_token = None

    tokens = m.get("tokens") or []
    if len(tokens) >= 2:
        for t in tokens:
            outcome = (t.get("outcome") or "").lower()
            symbol = (t.get("symbol") or "").lower()

            if "up" in outcome or "up" in symbol or "yes" in outcome:
                up_token = t.get("token_id") or t.get("id")
            elif "down" in outcome or "down" in symbol or "no" in outcome:
                down_token = t.get("token_id") or t.get("id")

        if not up_token:
            up_token = tokens[0].get("token_id") or tokens[0].get("id")
        if not down_token and len(tokens) > 1:
            down_token = tokens[1].get("token_id") or tokens[1].get("id")

    if not up_token or not down_token:
        up_token, down_token = _parse_clob_token_ids(m.get("clobTokenIds"))

    if not up_token or not down_token:
        return None

    start_ts = int(m.get("start_ts") or m.get("startTime") or 0)
    end_ts = int(m.get("end_ts") or m.get("endTime") or 0)

    return BtcUpDownMarket(
        event_id=str(event.get("id")),
        market_id=str(m.get("id")),
        up_token_id=str(up_token),
        down_token_id=str(down_token),
        title=str(title),
        slug=slug,
        start_ts=start_ts,
        end_ts=end_ts,
    )

def find_current_btc_updown_market() -> Optional[BtcUpDownMarket]:
    """
    1) If BTC_AGENT_MARKET_SLUG is set, try that first (debugging / backtesting).
    2) Otherwise, compute the slug for the current 5-minute window.
    3) If override fails, fall back to dynamic slug.
    """
    trading_cfg = get_trading_config()
    override_slug = trading_cfg.market_slug_override or os.getenv("BTC_AGENT_MARKET_SLUG")

    # 1. Try override slug if provided
    if override_slug:
        try:
            event = _fetch_event_by_slug(override_slug)
            market = _extract_market_from_event(event, override_slug)
            if market:
                return market
        except requests.HTTPError:
            pass  # fall through to dynamic

    # 2. Use dynamic current window slug
    slug = _current_btc_5m_slug()
    try:
        event = _fetch_event_by_slug(slug)
        market = _extract_market_from_event(event, slug)
        if market:
            return market
    except requests.HTTPError:
        return None

    return None
