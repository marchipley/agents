# custom/btc_agent/market_lookup.py

import json
import os
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from html import unescape
from typing import Optional

from .config import get_polymarket_config, get_trading_config
from .network import http_get


_SETTLEMENT_THRESHOLD_CACHE: dict[str, float] = {}
_MARKET_CACHE: dict[str, "BtcUpDownMarket"] = {}
_BTC_LIVE_PERIOD_OPEN_ATTEMPTS = 3
_BTC_LIVE_PERIOD_OPEN_RETRY_DELAY_SECONDS = 1.0
_NEXT_DATA_CHAIN_MAX_PAGES = 3
_NEXT_DATA_CHAIN_INTER_REQUEST_DELAY_SECONDS = 1.0


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
    volume: Optional[float] = None

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


def _extract_embedded_next_data_payload(html: str) -> Optional[dict]:
    next_data_match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json"(?:\s+crossorigin="anonymous")?>(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not next_data_match:
        return None

    try:
        payload = json.loads(next_data_match.group(1))
    except Exception:
        return None

    return payload if isinstance(payload, dict) else None


def _extract_build_id_from_payload(payload: object) -> Optional[str]:
    if isinstance(payload, dict):
        build_id = payload.get("buildId")
        if isinstance(build_id, str) and build_id:
            return build_id
        for value in payload.values():
            nested_build_id = _extract_build_id_from_payload(value)
            if nested_build_id:
                return nested_build_id

    if isinstance(payload, list):
        for value in payload:
            nested_build_id = _extract_build_id_from_payload(value)
            if nested_build_id:
                return nested_build_id

    return None


def _fetch_event_from_next_data_route(slug: str, build_id: str) -> Optional[dict]:
    url = f"https://polymarket.com/_next/data/{build_id}/en/event/{slug}.json"
    resp = http_get(url, params={"slug": slug}, timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    payload = resp.json()
    return _extract_event_from_next_data(payload, slug)


def _apply_threshold_from_next_data_payload(
    market: BtcUpDownMarket,
    slug: str,
    payload: dict,
) -> bool:
    live_period_open = _extract_live_period_open_from_next_data(payload, slug)
    if live_period_open is not None:
        market.settlement_threshold = live_period_open
        return True

    current_period_open = _extract_current_period_open_from_next_data(payload, slug)
    if current_period_open is not None:
        market.settlement_threshold = current_period_open
        return True

    previous_period_close = _extract_previous_period_close_from_next_data(
        payload,
        slug,
        allow_latest_prior_fallback=False,
    )
    if previous_period_close is not None:
        market.settlement_threshold = previous_period_close
        return True

    return False


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


def _extract_current_period_open_from_next_data(payload: dict, slug: str) -> Optional[float]:
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
        state = query.get("state") or {}
        data = state.get("data") or {}
        if not isinstance(data, dict):
            continue

        open_price = _coerce_btc_threshold(data.get("openPrice"))
        close_price = data.get("closePrice")
        if open_price is None or close_price not in (None, ""):
            continue

        query_key = query.get("queryKey") or []
        if (
            isinstance(query_key, list)
            and len(query_key) >= 4
            and _coerce_timestamp(query_key[3]) == current_start_ts
        ):
            return open_price

    for query in queries:
        state = query.get("state") or {}
        data = state.get("data") or {}
        if not isinstance(data, dict):
            continue

        open_price = _coerce_btc_threshold(data.get("openPrice"))
        close_price = data.get("closePrice")
        if open_price is not None and close_price in (None, ""):
            return open_price

    return None


def _extract_previous_period_close_from_next_data(
    payload: dict,
    slug: str,
    *,
    allow_latest_prior_fallback: bool = True,
) -> Optional[float]:
    slug_match = re.search(r"btc-updown-5m-(\d+)$", slug)
    if not slug_match:
        return None
    current_start_ts = int(slug_match.group(1))
    best_match: Optional[float] = None
    best_match_end_ts: Optional[int] = None

    def _walk(node) -> Optional[float]:
        nonlocal best_match
        nonlocal best_match_end_ts
        if isinstance(node, dict):
            results = node.get("results")
            if isinstance(results, list):
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    close_price = _coerce_btc_threshold(item.get("closePrice"))
                    end_ts = _coerce_timestamp(item.get("endTime") or item.get("endDate"))
                    if close_price is None or end_ts is None or end_ts > current_start_ts:
                        continue

                    if end_ts == current_start_ts:
                        return close_price

                    if allow_latest_prior_fallback and (
                        best_match_end_ts is None or end_ts > best_match_end_ts
                    ):
                        best_match = close_price
                        best_match_end_ts = end_ts

            for value in node.values():
                match = _walk(value)
                if match is not None:
                    return match

        if isinstance(node, list):
            for value in node:
                match = _walk(value)
                if match is not None:
                    return match

        return None

    exact_match = _walk(payload)
    if exact_match is not None:
        return exact_match
    return best_match


def _extract_previous_period_final_price_from_next_data(
    payload: dict,
    slug: str,
    *,
    allow_latest_prior_fallback: bool = True,
) -> Optional[float]:
    slug_match = re.search(r"btc-updown-5m-(\d+)$", slug)
    if not slug_match:
        return None
    current_start_ts = int(slug_match.group(1))
    best_match: Optional[float] = None
    best_match_end_ts: Optional[int] = None

    def _walk(node) -> Optional[float]:
        nonlocal best_match
        nonlocal best_match_end_ts
        if isinstance(node, dict):
            event_metadata = node.get("eventMetadata")
            if isinstance(event_metadata, dict):
                final_price = _coerce_btc_threshold(event_metadata.get("finalPrice"))
                end_ts = _coerce_timestamp(node.get("endTime") or node.get("endDate"))
                if final_price is not None and end_ts is not None and end_ts <= current_start_ts:
                    if end_ts == current_start_ts:
                        return final_price
                    if allow_latest_prior_fallback and (
                        best_match_end_ts is None or end_ts > best_match_end_ts
                    ):
                        best_match = final_price
                        best_match_end_ts = end_ts

            for value in node.values():
                match = _walk(value)
                if match is not None:
                    return match

        if isinstance(node, list):
            for value in node:
                match = _walk(value)
                if match is not None:
                    return match

        return None

    exact_match = _walk(payload)
    if exact_match is not None:
        return exact_match
    return best_match


def _fetch_next_data_payload(
    slug: str,
    build_id: str,
    *,
    request_number: int = 1,
) -> Optional[dict]:
    url = f"https://polymarket.com/_next/data/{build_id}/en/event/{slug}.json"
    resp = http_get(
        url,
        params={
            "slug": slug,
            "_req": request_number,
            "_ts": int(time.time() * 1000),
        },
        headers={
            "accept": "*/*",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "x-nextjs-data": "1",
        },
        timeout=10,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _fetch_next_data_payload_chain(
    slug: str,
    initial_build_id: str,
    *,
    max_pages: int = _NEXT_DATA_CHAIN_MAX_PAGES,
) -> list[tuple[str, Optional[dict]]]:
    pages: list[tuple[str, Optional[dict]]] = []
    build_id = initial_build_id

    while build_id and len(pages) < max_pages:
        if pages:
            time.sleep(_NEXT_DATA_CHAIN_INTER_REQUEST_DELAY_SECONDS)
        payload = _fetch_next_data_payload(
            slug,
            build_id,
            request_number=len(pages) + 1,
        )
        pages.append((build_id, payload))
        if not isinstance(payload, dict):
            break
        next_build_id = _extract_build_id_from_payload(payload)
        if next_build_id:
            build_id = next_build_id

    return pages


def _build_current_period_dataset(
    slug: str,
    html: str,
    embedded_payload: Optional[dict],
    build_id: Optional[str],
    payload_chain: list[tuple[str, Optional[dict]]],
) -> dict:
    page_url = f"https://polymarket.com/event/{slug}"
    selected_next_data_pages = []
    for request_number, (page_build_id, payload) in enumerate(payload_chain, start=1):
        if request_number not in {1, 2, 3}:
            continue
        selected_next_data_pages.append(
            {
                "request_number": request_number,
                "build_id": page_build_id,
                "next_data_url": f"https://polymarket.com/_next/data/{page_build_id}/en/event/{slug}.json?slug={slug}",
                "payload": payload,
            }
        )

    return {
        "slug": slug,
        "page_url": page_url,
        "build_id": build_id,
        "selected_next_data_pages": selected_next_data_pages,
    }


def _write_current_period_dataset_file(dataset: dict) -> None:
    data_dir = os.path.join(os.getcwd(), "data_files")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "current_period.json")
    with open(output_path, "w", encoding="utf-8") as data_file:
        json.dump(dataset, data_file, indent=2, sort_keys=True)


def build_price_to_beat_debug_report(slug: str) -> str:
    reports = build_price_to_beat_debug_reports(slug)
    return reports[0] if reports else ""


def build_price_to_beat_debug_reports(slug: str) -> list[str]:
    page_lines = [f"slug={slug}"]
    page_url = f"https://polymarket.com/event/{slug}"
    page_lines.append(f"page_url={page_url}")
    reports: list[str] = []

    try:
        html = _fetch_polymarket_page(slug)
        page_lines.append("page_fetch=success")
    except Exception as exc:
        page_lines.append(f"page_fetch=error: {exc}")
        return ["\n".join(page_lines) + "\n"]

    embedded_payload = _extract_embedded_next_data_payload(html)
    page_lines.append(
        f"live_period_open={_extract_live_period_open_from_next_data(embedded_payload, slug) if isinstance(embedded_payload, dict) else None}"
    )
    page_lines.append(
        f"current_period_open={_extract_current_period_open_from_next_data(embedded_payload, slug) if isinstance(embedded_payload, dict) else None}"
    )
    page_lines.append(
        "previous_period_close_from_results="
        + str(
            _extract_previous_period_close_from_next_data(
                embedded_payload,
                slug,
                allow_latest_prior_fallback=True,
            )
            if isinstance(embedded_payload, dict)
            else None
        )
    )
    page_lines.append(
        "previous_period_final_price_from_event_metadata="
        + str(
            _extract_previous_period_final_price_from_next_data(
                embedded_payload,
                slug,
                allow_latest_prior_fallback=True,
            )
            if isinstance(embedded_payload, dict)
            else None
        )
    )
    page_lines.append("embedded_page_payload=")
    page_lines.append(json.dumps(embedded_payload, indent=2, sort_keys=True) if isinstance(embedded_payload, dict) else "None")

    build_id = _extract_next_build_id(html)
    page_lines.append(f"build_id={build_id or 'None'}")
    reports.append("\n".join(page_lines) + "\n")
    if not build_id:
        return reports

    first_next_data_url = (
        f"https://polymarket.com/_next/data/{build_id}/en/event/{slug}.json?slug={slug}"
    )
    page_lines.append(f"next_data_curl=curl '{first_next_data_url}'")
    reports[0] = "\n".join(page_lines) + "\n"

    try:
        payload_chain = _fetch_next_data_payload_chain(slug, build_id)
    except Exception as exc:
        reports.append(
            "\n".join(
                [
                    f"slug={slug}",
                    f"page_url={page_url}",
                    f"build_id={build_id}",
                    f"next_data_curl=curl 'https://polymarket.com/_next/data/{build_id}/en/event/{slug}.json?slug={slug}'",
                    f"next_data_fetch=error: {exc}",
                ]
            )
            + "\n"
        )
        return reports

    for fetched_build_id, payload in payload_chain:
        next_data_url = (
            f"https://polymarket.com/_next/data/{fetched_build_id}/en/event/{slug}.json?slug={slug}"
        )
        next_data_lines = [
            f"slug={slug}",
            f"page_url={page_url}",
            f"build_id={fetched_build_id}",
            f"next_data_curl=curl '{next_data_url}'",
        ]
        if isinstance(payload, dict):
            next_data_lines.append("next_data_fetch=success")
            next_data_lines.append(
                f"live_period_open={_extract_live_period_open_from_next_data(payload, slug)}"
            )
            next_data_lines.append(
                f"current_period_open={_extract_current_period_open_from_next_data(payload, slug)}"
            )
            next_data_lines.append(
                "previous_period_close_from_results="
                + str(
                    _extract_previous_period_close_from_next_data(
                        payload,
                        slug,
                        allow_latest_prior_fallback=True,
                    )
                )
            )
            next_data_lines.append(
                "previous_period_final_price_from_event_metadata="
                + str(
                    _extract_previous_period_final_price_from_next_data(
                        payload,
                        slug,
                        allow_latest_prior_fallback=True,
                    )
                )
            )
            event = _extract_event_from_next_data(payload, slug)
            if isinstance(event, dict):
                market = (event.get("markets") or [{}])[0]
                next_data_lines.append(
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
                next_data_lines.append("event_threshold=None")
            next_data_lines.append("next_data_payload=")
            next_data_lines.append(json.dumps(payload, indent=2, sort_keys=True))
        else:
            next_data_lines.append("next_data_fetch=error: payload was None")
        reports.append("\n".join(next_data_lines) + "\n")

    return reports


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


def _extract_vatic_price_from_response(payload) -> Optional[float]:
    if isinstance(payload, (int, float, str)):
        return _coerce_btc_threshold(payload)

    if not isinstance(payload, dict):
        return None

    threshold = _coerce_btc_threshold(payload.get("price"))
    if threshold is not None:
        return threshold

    for nested_key in ("data", "result", "target"):
        nested_payload = payload.get(nested_key)
        if nested_payload is None:
            continue
        threshold = _extract_vatic_price_from_response(nested_payload)
        if threshold is not None:
            return threshold

    return None


def _fetch_vatic_price_to_beat_by_slug(slug: str) -> Optional[float]:
    slug_match = re.search(r"btc-updown-5m-(\d+)$", slug)
    if not slug_match:
        return None

    timestamp = slug_match.group(1)
    resp = http_get(
        "https://api.vatic.trading/api/v1/targets/timestamp",
        params={
            "asset": "btc",
            "type": "5min",
            "timestamp": timestamp,
        },
        timeout=10,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    try:
        payload = resp.json()
    except ValueError:
        return _coerce_btc_threshold(resp.text.strip())

    return _extract_vatic_price_from_response(payload)


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
    volume = _coerce_threshold(m.get("volume"))

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
        volume=volume,
    )


def _hydrate_missing_threshold_from_page(market: Optional[BtcUpDownMarket], slug: str) -> Optional[BtcUpDownMarket]:
    if market is None:
        return market

    if slug.startswith("btc-updown-5m-"):
        try:
            vatic_threshold = _fetch_vatic_price_to_beat_by_slug(slug)
        except Exception:
            vatic_threshold = None
        if vatic_threshold is not None:
            market.settlement_threshold = vatic_threshold
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

    embedded_payload = _extract_embedded_next_data_payload(html)

    build_id = _extract_next_build_id(html)
    payload_chain: list[tuple[str, Optional[dict]]] = []
    if build_id:
        next_data_attempts = 1
        if slug.startswith("btc-updown-5m-"):
            next_data_attempts = _BTC_LIVE_PERIOD_OPEN_ATTEMPTS

        for attempt in range(next_data_attempts):
            try:
                candidate_payload_chain = _fetch_next_data_payload_chain(slug, build_id)
            except Exception:
                candidate_payload_chain = []
            if candidate_payload_chain:
                payload_chain = candidate_payload_chain

            if attempt < next_data_attempts - 1:
                time.sleep(_BTC_LIVE_PERIOD_OPEN_RETRY_DELAY_SECONDS)

        dataset = _build_current_period_dataset(
            slug=slug,
            html=html,
            embedded_payload=embedded_payload if isinstance(embedded_payload, dict) else None,
            build_id=build_id,
            payload_chain=payload_chain,
        )
        _write_current_period_dataset_file(dataset)

        if isinstance(embedded_payload, dict) and _apply_threshold_from_next_data_payload(
            market,
            slug,
            embedded_payload,
        ):
            return market

        for _, next_data_payload in payload_chain:
            if isinstance(next_data_payload, dict) and _apply_threshold_from_next_data_payload(
                market,
                slug,
                next_data_payload,
            ):
                return market

        for _, next_data_payload in payload_chain:
            next_data_event = (
                _extract_event_from_next_data(next_data_payload, slug)
                if isinstance(next_data_payload, dict)
                else None
            )
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

    dataset = _build_current_period_dataset(
        slug=slug,
        html=html,
        embedded_payload=embedded_payload if isinstance(embedded_payload, dict) else None,
        build_id=build_id,
        payload_chain=payload_chain,
    )
    _write_current_period_dataset_file(dataset)

    if isinstance(embedded_payload, dict) and _apply_threshold_from_next_data_payload(
        market,
        slug,
        embedded_payload,
    ):
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
        _MARKET_CACHE[market.slug] = replace(market)
    return market


def get_btc_updown_market_by_slug(slug: str) -> Optional[BtcUpDownMarket]:
    cached_market = _MARKET_CACHE.get(slug)
    if cached_market is not None:
        return replace(cached_market)

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
