# custom/btc_agent/market_lookup.py

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .config import get_polymarket_config, get_trading_config
from .network import http_get

@dataclass
class BtcUpDownMarket:
    event_id: str
    market_id: str
    up_token_id: str
    down_token_id: str
    title: str
    question: str
    slug: str
    start_ts: int
    end_ts: int
    settlement_threshold: Optional[float]

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
    resp = http_get(url, timeout=10)
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


def _coerce_threshold(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_threshold_from_text(*values: Optional[str]) -> Optional[float]:
    patterns = [
        r"(?:finish|ends?|closes?|settles?)\s+(?:above|below|at|over|under)\s+\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"(?:above|below|at|over|under)\s+\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"target(?:\s+price)?\s*(?:is|:)?\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    ]

    for value in values:
        if not value:
            continue
        for pattern in patterns:
            match = re.search(pattern, value, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    continue
    return None

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
    question = str(m.get("question") or "")
    settlement_threshold = (
        _coerce_threshold(m.get("groupItemThreshold"))
        or _coerce_threshold(m.get("threshold"))
        or _parse_threshold_from_text(
            question,
            str(m.get("description") or ""),
            title,
        )
    )

    return BtcUpDownMarket(
        event_id=str(event.get("id")),
        market_id=str(m.get("id")),
        up_token_id=str(up_token),
        down_token_id=str(down_token),
        title=str(title),
        question=question,
        slug=slug,
        start_ts=start_ts,
        end_ts=end_ts,
        settlement_threshold=settlement_threshold,
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
