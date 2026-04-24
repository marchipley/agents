# custom/btc_agent/market_lookup.py

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from typing import Optional

from .config import get_polymarket_config, get_trading_config
from .network import http_get


_SETTLEMENT_THRESHOLD_CACHE: dict[str, float] = {}


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
        coerced = float(value)
        return coerced if coerced != 0 else None
    except (TypeError, ValueError):
        return None


def _coerce_btc_threshold(value) -> Optional[float]:
    threshold = _coerce_threshold(value)
    if threshold is None:
        return None
    return threshold if threshold >= 1000 else None


def _coerce_timestamp(value) -> int:
    if value in (None, ""):
        return 0

    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip()
    if not text:
        return 0

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


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
                    threshold = float(match.group(1).replace(",", ""))
                    if threshold >= 1000:
                        return threshold
                except ValueError:
                    continue
    return None


def _extract_settlement_threshold(
    event: dict,
    market: dict,
    title: str,
    question: str,
    slug: Optional[str] = None,
) -> Optional[float]:
    if slug and slug.startswith("btc-updown-5m-"):
        return _parse_threshold_from_text(
            question,
            str(market.get("description") or ""),
            title,
        )

    event_metadata = event.get("eventMetadata") or {}
    return (
        _coerce_btc_threshold(event_metadata.get("priceToBeat"))
        or _coerce_btc_threshold(event_metadata.get("price_to_beat"))
        or _coerce_btc_threshold(event_metadata.get("price_to_beat_usd"))
        or _coerce_btc_threshold(market.get("groupItemThreshold"))
        or _coerce_btc_threshold(market.get("threshold"))
        or _parse_threshold_from_text(
            question,
            str(market.get("description") or ""),
            title,
        )
    )


def _extract_event_from_next_data(payload: dict, slug: str) -> Optional[dict]:
    page_props = payload.get("pageProps")
    if not isinstance(page_props, dict):
        props = payload.get("props") or {}
        page_props = props.get("pageProps") or {}
    dehydrated_state = page_props.get("dehydratedState") or {}
    queries = dehydrated_state.get("queries") or []

    for query in queries:
        query_key = query.get("queryKey") or []
        if (
            isinstance(query_key, list)
            and len(query_key) >= 2
            and query_key[0] == "/api/event/slug"
            and query_key[1] == slug
        ):
            state = query.get("state") or {}
            data = state.get("data")
            if isinstance(data, dict):
                return data

    event = page_props.get("event")
    if isinstance(event, dict):
        return event

    market = page_props.get("market")
    if isinstance(market, dict):
        return market
    return None


def _extract_next_build_id(html: str) -> Optional[str]:
    next_data_match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json"(?:\s+crossorigin="anonymous")?>(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if next_data_match:
        try:
            payload = json.loads(next_data_match.group(1))
            build_id = payload.get("buildId")
            if isinstance(build_id, str) and build_id:
                return build_id
        except Exception:
            pass

    match = re.search(r'"buildId"\s*:\s*"([^"]+)"', html)
    if not match:
        return None
    return match.group(1)


def _fetch_event_from_next_data_route(slug: str, build_id: str) -> Optional[dict]:
    url = f"https://polymarket.com/_next/data/{build_id}/en/event/{slug}.json"
    resp = http_get(url, params={"slug": slug}, timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    payload = resp.json()
    return _extract_event_from_next_data(payload, slug)


def _extract_live_period_open_from_next_data(payload: dict, slug: str) -> Optional[float]:
    page_props = payload.get("pageProps")
    if not isinstance(page_props, dict):
        props = payload.get("props") or {}
        page_props = props.get("pageProps") or {}

    dehydrated_state = page_props.get("dehydratedState") or {}
    queries = dehydrated_state.get("queries") or []

    slug_match = re.search(r"btc-updown-5m-(\d+)$", slug)
    if not slug_match:
        return None
    current_start_ts = int(slug_match.group(1))

    for query in queries:
        query_key = query.get("queryKey") or []
        if not (
            isinstance(query_key, list)
            and len(query_key) >= 6
            and query_key[0] == "crypto-prices"
            and query_key[1] == "price"
            and str(query_key[2]).upper() == "BTC"
            and str(query_key[4]).lower() == "fiveminute"
        ):
            continue

        start_ts = _coerce_timestamp(query_key[3])
        if start_ts != current_start_ts:
            continue

        state = query.get("state") or {}
        data = state.get("data") or {}
        if not isinstance(data, dict):
            continue

        open_price = _coerce_btc_threshold(data.get("openPrice"))
        if open_price is not None:
            return open_price

    return None


def _extract_previous_period_close_from_next_data(payload: dict, slug: str) -> Optional[float]:
    page_props = payload.get("pageProps")
    if not isinstance(page_props, dict):
        props = payload.get("props") or {}
        page_props = props.get("pageProps") or {}

    dehydrated_state = page_props.get("dehydratedState") or {}
    queries = dehydrated_state.get("queries") or []

    slug_match = re.search(r"btc-updown-5m-(\d+)$", slug)
    if not slug_match:
        return None
    current_start_ts = int(slug_match.group(1))

    best_match = None
    for query in queries:
        state = query.get("state") or {}
        data = state.get("data") or {}
        inner_data = data.get("data") if isinstance(data, dict) else None
        results = inner_data.get("results") if isinstance(inner_data, dict) else None
        if not isinstance(results, list):
            continue

        for item in results:
            if not isinstance(item, dict):
                continue
            close_price = _coerce_btc_threshold(item.get("closePrice"))
            end_time = item.get("endTime")
            if close_price is None or not end_time:
                continue

            end_ts = _coerce_timestamp(end_time)
            if end_ts != current_start_ts:
                continue

            best_match = close_price

    return best_match


def _fetch_next_data_payload(slug: str, build_id: str) -> Optional[dict]:
    url = f"https://polymarket.com/_next/data/{build_id}/en/event/{slug}.json"
    resp = http_get(url, params={"slug": slug}, timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def build_price_to_beat_debug_report(slug: str) -> str:
    lines = [f"slug={slug}"]
    page_url = f"https://polymarket.com/event/{slug}"
    lines.append(f"page_url={page_url}")

    try:
        html = _fetch_polymarket_page(slug)
        lines.append("page_fetch=success")
    except Exception as exc:
        lines.append(f"page_fetch=error: {exc}")
        return "\n".join(lines) + "\n"

    build_id = _extract_next_build_id(html)
    lines.append(f"build_id={build_id or 'None'}")
    if not build_id:
        lines.append("next_data_curl=None")
        return "\n".join(lines) + "\n"

    next_data_url = f"https://polymarket.com/_next/data/{build_id}/en/event/{slug}.json?slug={slug}"
    lines.append(f"next_data_curl=curl '{next_data_url}'")

    try:
        payload = _fetch_next_data_payload(slug, build_id)
        lines.append("next_data_fetch=success")
        lines.append(
            f"live_period_open={_extract_live_period_open_from_next_data(payload, slug)}"
        )
        lines.append(
            f"previous_period_close_from_results={_extract_previous_period_close_from_next_data(payload, slug)}"
        )
        event = _extract_event_from_next_data(payload, slug)
        if isinstance(event, dict):
            market = (event.get("markets") or [{}])[0]
            lines.append(
                "event_threshold="
                + str(
                    _extract_settlement_threshold(
                        event,
                        market if isinstance(market, dict) else {},
                        str(event.get("title") or ""),
                        str((market or {}).get("question") or "")
                        if isinstance(market, dict)
                        else "",
                    )
                )
            )
        else:
            lines.append("event_threshold=None")
        lines.append("next_data_payload=")
        lines.append(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as exc:
        lines.append(f"next_data_fetch=error: {exc}")

    return "\n".join(lines) + "\n"


def _extract_threshold_from_price_to_beat_response(payload) -> Optional[float]:
    if isinstance(payload, (int, float, str)):
        return _coerce_btc_threshold(payload)

    if not isinstance(payload, dict):
        return None

    candidate_fields = (
        "priceToBeat",
        "price_to_beat",
        "price_to_beat_usd",
        "price",
        "value",
    )
    for field in candidate_fields:
        threshold = _coerce_btc_threshold(payload.get(field))
        if threshold is not None:
            return threshold

    for nested_key in ("data", "result"):
        nested_payload = payload.get(nested_key)
        if nested_payload is None:
            continue
        threshold = _extract_threshold_from_price_to_beat_response(nested_payload)
        if threshold is not None:
            return threshold

    return None


def _fetch_price_to_beat_by_slug(slug: str) -> Optional[float]:
    if slug.startswith("btc-updown-5m-"):
        return None

    resp = http_get(f"https://polymarket.com/api/equity/price-to-beat/{slug}", timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    try:
        payload = resp.json()
    except ValueError:
        return _coerce_btc_threshold(resp.text.strip())

    return _extract_threshold_from_price_to_beat_response(payload)


def _extract_threshold_from_page_html(html: str) -> Optional[float]:
    decoded_html = unescape(html)

    direct_span_match = re.search(
        r"<span[^>]*class=['\"][^'\"]*\btext-text-secondary\b[^'\"]*\btext-heading-2xl\b[^'\"]*['\"][^>]*>\s*\$([0-9][0-9,]*(?:\.[0-9]+)?)\s*</span>",
        decoded_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if direct_span_match:
        try:
            return _coerce_btc_threshold(direct_span_match.group(1).replace(",", ""))
        except ValueError:
            pass

    labeled_span_match = re.search(
        r"Price\s+To\s+Beat.*?<span[^>]*class=\"[^\"]*text-text-secondary[^\"]*text-heading-2xl[^\"]*\"[^>]*>\s*\$([0-9][0-9,]*(?:\.[0-9]+)?)\s*</span>",
        decoded_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if labeled_span_match:
        try:
            return _coerce_btc_threshold(labeled_span_match.group(1).replace(",", ""))
        except ValueError:
            pass

    label_match = re.search(r"Price\s+To\s+Beat|Price\s+to\s+Beat", decoded_html, flags=re.IGNORECASE)
    if label_match:
        trailing_html = decoded_html[label_match.end() : label_match.end() + 2000]
        price_match = re.search(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)", trailing_html)
        if price_match:
            try:
                return _coerce_btc_threshold(price_match.group(1).replace(",", ""))
            except ValueError:
                pass

    patterns = [
        r"Price to Beat\s*\(\$([0-9][0-9,]*(?:\.[0-9]+)?)\s*\)",
        r'Price to Beat"\s+of\s+\$([0-9][0-9,]*(?:\.[0-9]+)?)',
        r'Price to Beat"\s*\(\$([0-9][0-9,]*(?:\.[0-9]+)?)\s*\)',
    ]
    for pattern in patterns:
        match = re.search(pattern, decoded_html, flags=re.IGNORECASE)
        if match:
            try:
                return _coerce_btc_threshold(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _fetch_polymarket_page(slug: str) -> str:
    url = f"https://polymarket.com/event/{slug}"
    resp = http_get(url, timeout=10)
    resp.raise_for_status()
    return resp.text


def _fetch_event_from_polymarket_page(slug: str) -> Optional[dict]:
    html = _fetch_polymarket_page(slug)
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json" crossorigin="anonymous">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        return None

    payload = json.loads(match.group(1))
    return _extract_event_from_next_data(payload, slug)

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

    start_ts = _coerce_timestamp(
        m.get("start_ts")
        or m.get("startTime")
        or m.get("eventStartTime")
        or m.get("startDate")
    )
    end_ts = _coerce_timestamp(
        m.get("end_ts")
        or m.get("endTime")
        or m.get("umaEndDate")
        or m.get("endDate")
    )
    question = str(m.get("question") or "")
    settlement_threshold = _extract_settlement_threshold(event, m, title, question, slug=slug)

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


def _hydrate_missing_threshold_from_page(market: Optional[BtcUpDownMarket], slug: str) -> Optional[BtcUpDownMarket]:
    if market is None:
        return market

    try:
        api_threshold = _fetch_price_to_beat_by_slug(slug)
    except Exception:
        api_threshold = None
    if api_threshold is not None:
        market.settlement_threshold = api_threshold
        return market

    try:
        html = _fetch_polymarket_page(slug)
    except Exception:
        return market

    build_id = _extract_next_build_id(html)
    if build_id:
        try:
            next_data_payload = _fetch_next_data_payload(slug, build_id)
        except Exception:
            next_data_payload = None
        if isinstance(next_data_payload, dict):
            previous_period_close = _extract_previous_period_close_from_next_data(next_data_payload, slug)
            if previous_period_close is not None:
                market.settlement_threshold = previous_period_close
                return market

            live_period_open = _extract_live_period_open_from_next_data(next_data_payload, slug)
            if live_period_open is not None:
                market.settlement_threshold = live_period_open
                return market

            next_data_event = _extract_event_from_next_data(next_data_payload, slug)
        else:
            next_data_event = None
        if isinstance(next_data_event, dict):
            next_data_threshold = _extract_settlement_threshold(
                next_data_event,
                (next_data_event.get("markets") or [{}])[0],
                str(next_data_event.get("title") or market.title),
                str(((next_data_event.get("markets") or [{}])[0]).get("question") or market.question),
            )
            if next_data_threshold is not None:
                market.settlement_threshold = next_data_threshold
                return market

        if slug.startswith("btc-updown-5m-"):
            return market

    page_threshold = _extract_threshold_from_page_html(html)
    if page_threshold is not None:
        market.settlement_threshold = page_threshold
        return market

    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json" crossorigin="anonymous">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        return market

    payload = json.loads(match.group(1))
    page_event = _extract_event_from_next_data(payload, slug)
    if not page_event:
        return market

    hydrated_market = _extract_market_from_event(page_event, slug)
    if hydrated_market and hydrated_market.settlement_threshold is not None:
        return hydrated_market

    return market


def _apply_cached_settlement_threshold(market: Optional[BtcUpDownMarket]) -> Optional[BtcUpDownMarket]:
    if market is None:
        return None
    cached_threshold = _SETTLEMENT_THRESHOLD_CACHE.get(market.slug)
    if cached_threshold is not None:
        market.settlement_threshold = cached_threshold
    return market


def _cache_settlement_threshold(market: Optional[BtcUpDownMarket]) -> Optional[BtcUpDownMarket]:
    if market is None:
        return None
    if _coerce_btc_threshold(market.settlement_threshold) is not None:
        _SETTLEMENT_THRESHOLD_CACHE[market.slug] = float(market.settlement_threshold)
    return market


def get_btc_updown_market_by_slug(slug: str) -> Optional[BtcUpDownMarket]:
    event = _fetch_event_by_slug(slug)
    market = _extract_market_from_event(event, slug)
    if market is None:
        return None
    market = _apply_cached_settlement_threshold(market)
    if _coerce_btc_threshold(market.settlement_threshold) is not None:
        return _cache_settlement_threshold(market)
    market = _hydrate_missing_threshold_from_page(market, slug)
    return _cache_settlement_threshold(market)

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
            market = get_btc_updown_market_by_slug(override_slug)
            if market:
                return market
        except requests.HTTPError:
            pass  # fall through to dynamic

    # 2. Use dynamic current window slug
    slug = _current_btc_5m_slug()
    try:
        market = get_btc_updown_market_by_slug(slug)
        if market:
            return market
    except requests.HTTPError:
        return None

    return None
