# custom/btc_agent/indicators.py

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
import time
from typing import Optional, List, Tuple
import statistics

import requests
try:
    import websocket
except ImportError:  # pragma: no cover
    websocket = None

from .network import http_get

# Simple in-memory history: list of (timestamp, price)
_PRICE_HISTORY: List[Tuple[datetime, float]] = []
_LAST_SUCCESSFUL_PROVIDER_INDEX = 0
_PRICE_HISTORY_BACKFILLED = False
_BACKFILL_WINDOW_SECONDS = 300
_BACKFILL_BUCKET_SECONDS = 20
_WINDOW_BASELINE_CARRY_FORWARD_SECONDS = 60
_WINDOW_BASELINE_LOOKAHEAD_SECONDS = 60
_BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
_BINANCE_WS_TIMEOUT_SECONDS = 3.0
_POLYMARKET_RTDS_URL = "wss://ws-live-data.polymarket.com"
_POLYMARKET_RTDS_SYMBOL = "btcusdt"
_POLYMARKET_RTDS_FILTERS = json.dumps({"symbol": _POLYMARKET_RTDS_SYMBOL})
_POLYMARKET_RTDS_TIMEOUT_SECONDS = 3.0
_POLYMARKET_RTDS_MAX_MESSAGES = 8
_POLYMARKET_RTDS_MAX_SNAPSHOT_AGE_SECONDS = 3.0


@dataclass
class BtcFeatures:
    as_of: datetime
    price_usd: float
    window_open_price: float
    trailing_5m_open_price: float
    delta_pct_from_window_open: float
    delta_pct_from_trailing_5m_open: float
    delta_from_previous_tick: Optional[float]
    rsi_14: Optional[float]
    momentum_1m: Optional[float]
    momentum_5m: Optional[float]
    velocity_15s: Optional[float]
    velocity_30s: Optional[float]
    volatility_5m: Optional[float]
    consecutive_flat_ticks: int
    retained_sample_count: int
    window_sample_count: int
    trailing_5m_sample_count: int


def _fetch_spot_price_from_coingecko() -> float:
    resp = http_get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": "bitcoin", "vs_currencies": "usd"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return float(data["bitcoin"]["usd"])


def _fetch_spot_price_from_coinbase() -> float:
    resp = http_get(
        "https://api.coinbase.com/v2/prices/BTC-USD/spot",
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return float(data["data"]["amount"])


def _create_polymarket_rtds_connection():
    if websocket is None:
        raise requests.RequestException("websocket-client is not installed")

    proxy_env_names = (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    )
    saved_proxy_env = {name: os.environ.get(name) for name in proxy_env_names}
    try:
        for name in proxy_env_names:
            os.environ.pop(name, None)
        return websocket.create_connection(
            _POLYMARKET_RTDS_URL,
            timeout=_POLYMARKET_RTDS_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise requests.RequestException(f"Unable to connect to Polymarket RTDS: {exc}") from exc
    finally:
        for name, value in saved_proxy_env.items():
            if value is not None:
                os.environ[name] = value


def _create_binance_connection():
    if websocket is None:
        raise requests.RequestException("websocket-client is not installed")

    proxy_env_names = (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    )
    saved_proxy_env = {name: os.environ.get(name) for name in proxy_env_names}
    try:
        for name in proxy_env_names:
            os.environ.pop(name, None)
        return websocket.create_connection(
            _BINANCE_WS_URL,
            timeout=_BINANCE_WS_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise requests.RequestException(f"Unable to connect to Binance websocket: {exc}") from exc
    finally:
        for name, value in saved_proxy_env.items():
            if value is not None:
                os.environ[name] = value


def _parse_binance_ticker_price(message: dict) -> Optional[Tuple[float, datetime]]:
    symbol = str(message.get("s", "")).upper()
    if symbol != "BTCUSDT":
        return None
    last_price = message.get("c")
    event_time_ms = message.get("E")
    if last_price is None or event_time_ms is None:
        return None
    return (
        float(last_price),
        datetime.fromtimestamp(float(event_time_ms) / 1000, tz=timezone.utc),
    )


def _fetch_spot_price_from_binance_websocket() -> float:
    ws = None
    try:
        ws = _create_binance_connection()
        raw_message = ws.recv()
        if not raw_message:
            raise requests.RequestException("Binance websocket returned no data")
        message = json.loads(raw_message)
        parsed = _parse_binance_ticker_price(message)
        if parsed is None:
            raise requests.RequestException("Binance websocket returned an unexpected ticker payload")
        price, as_of = parsed
        _record_price_sample(price, as_of=as_of)
        return price
    except (
        OSError,
        ValueError,
        requests.RequestException,
    ) as exc:
        raise requests.RequestException(f"Binance websocket BTC price fetch failed: {exc}") from exc
    except Exception as exc:
        if websocket is not None and isinstance(exc, websocket.WebSocketException):
            raise requests.RequestException(f"Binance websocket BTC price fetch failed: {exc}") from exc
        raise
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


def _parse_rtds_snapshot_price(message: dict) -> Optional[Tuple[float, datetime]]:
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return None
    if str(payload.get("symbol", "")).lower() != _POLYMARKET_RTDS_SYMBOL:
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    latest_item = data[-1]
    if not isinstance(latest_item, dict):
        return None
    value = latest_item.get("value")
    timestamp_ms = latest_item.get("timestamp")
    if value is None or timestamp_ms is None:
        return None
    return (
        float(value),
        datetime.fromtimestamp(float(timestamp_ms) / 1000, tz=timezone.utc),
    )


def _parse_rtds_update_price(message: dict) -> Optional[Tuple[float, datetime]]:
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return None
    if str(payload.get("symbol", "")).lower() != _POLYMARKET_RTDS_SYMBOL:
        return None
    value = payload.get("value")
    timestamp_ms = payload.get("timestamp")
    if value is None or timestamp_ms is None:
        return None
    return (
        float(value),
        datetime.fromtimestamp(float(timestamp_ms) / 1000, tz=timezone.utc),
    )


def _fetch_spot_price_from_polymarket_rtds() -> float:
    ws = None
    latest_snapshot: Optional[Tuple[float, datetime]] = None
    subscribe_message = {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices",
                "type": "update",
                "filters": _POLYMARKET_RTDS_FILTERS,
            }
        ],
    }
    deadline = time.monotonic() + _POLYMARKET_RTDS_TIMEOUT_SECONDS

    try:
        ws = _create_polymarket_rtds_connection()
        ws.send(json.dumps(subscribe_message))

        for _ in range(_POLYMARKET_RTDS_MAX_MESSAGES):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            ws.settimeout(remaining)
            raw_message = ws.recv()
            if not raw_message or raw_message in ("PING", "PONG"):
                continue

            message = json.loads(raw_message)
            if message.get("topic") != "crypto_prices":
                continue

            if message.get("type") == "update":
                parsed_update = _parse_rtds_update_price(message)
                if parsed_update is not None:
                    price, as_of = parsed_update
                    _record_price_sample(price, as_of=as_of)
                    return price

            parsed_snapshot = _parse_rtds_snapshot_price(message)
            if parsed_snapshot is not None:
                latest_snapshot = parsed_snapshot

        if latest_snapshot is not None:
            price, as_of = latest_snapshot
            snapshot_age_seconds = max(
                (datetime.now(timezone.utc) - as_of).total_seconds(),
                0.0,
            )
            if snapshot_age_seconds <= _POLYMARKET_RTDS_MAX_SNAPSHOT_AGE_SECONDS:
                _record_price_sample(price, as_of=as_of)
                return price
            raise requests.RequestException(
                f"Polymarket RTDS snapshot was stale ({snapshot_age_seconds:.1f}s old)"
            )
    except (
        OSError,
        ValueError,
        requests.RequestException,
    ) as exc:
        raise requests.RequestException(f"Polymarket RTDS BTC price fetch failed: {exc}") from exc
    except Exception as exc:
        if websocket is not None and isinstance(exc, websocket.WebSocketException):
            raise requests.RequestException(f"Polymarket RTDS BTC price fetch failed: {exc}") from exc
        raise
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    raise requests.RequestException("Polymarket RTDS returned no fresh BTC price update")


def _get_price_providers():
    return [
        ("Polymarket RTDS", _fetch_spot_price_from_polymarket_rtds),
        ("Binance WebSocket", _fetch_spot_price_from_binance_websocket),
        ("Coinbase", _fetch_spot_price_from_coinbase),
        ("CoinGecko", _fetch_spot_price_from_coingecko),
    ]


def get_latest_cached_price() -> Optional[float]:
    if not _PRICE_HISTORY:
        return None
    return _PRICE_HISTORY[-1][1]


def _record_price_sample(price: float, as_of: Optional[datetime] = None) -> None:
    global _PRICE_HISTORY

    timestamp = as_of or datetime.now(timezone.utc)
    _PRICE_HISTORY.append((timestamp, price))
    if len(_PRICE_HISTORY) > 60:
        _PRICE_HISTORY = _PRICE_HISTORY[-60:]


def _fetch_recent_trades_from_coinbase(limit: int = 1000) -> List[Tuple[datetime, float]]:
    resp = http_get(
        "https://api.exchange.coinbase.com/products/BTC-USD/trades",
        params={"limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    payload = resp.json()
    trades: List[Tuple[datetime, float]] = []
    for item in payload:
        trade_time = item.get("time")
        price = item.get("price")
        if not trade_time or price is None:
            continue
        trades.append(
            (
                datetime.fromisoformat(str(trade_time).replace("Z", "+00:00")),
                float(price),
            )
        )
    trades.sort(key=lambda pair: pair[0])
    return trades


def _fetch_coinbase_candles(start: datetime, end: datetime) -> List[Tuple[datetime, float, float]]:
    resp = http_get(
        "https://api.exchange.coinbase.com/products/BTC-USD/candles",
        params={
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "granularity": 60,
        },
        timeout=10,
    )
    resp.raise_for_status()
    payload = resp.json()
    candles: List[Tuple[datetime, float, float]] = []
    for item in payload:
        if not isinstance(item, list) or len(item) < 5:
            continue
        candles.append(
            (
                datetime.fromtimestamp(int(item[0]), tz=timezone.utc),
                float(item[3]),
                float(item[4]),
            )
        )
    candles.sort(key=lambda item: item[0])
    return candles


def _seed_price_history_from_trades(now: datetime) -> bool:
    cutoff_ts = now.timestamp() - _BACKFILL_WINDOW_SECONDS
    trades = [
        (ts, price)
        for ts, price in _fetch_recent_trades_from_coinbase()
        if cutoff_ts <= ts.timestamp() <= now.timestamp()
    ]
    if not trades:
        return False

    trade_index = 0
    last_price: Optional[float] = None
    seeded_samples = 0

    for bucket_ts in range(int(cutoff_ts), int(now.timestamp()) + 1, _BACKFILL_BUCKET_SECONDS):
        bucket_time = datetime.fromtimestamp(bucket_ts, tz=timezone.utc)
        while trade_index < len(trades) and trades[trade_index][0] <= bucket_time:
            last_price = trades[trade_index][1]
            trade_index += 1

        if last_price is None:
            last_price = trades[0][1]

        _record_price_sample(last_price, as_of=bucket_time)
        seeded_samples += 1

    return seeded_samples >= 15


def _seed_price_history_from_candles(now: datetime) -> bool:
    start = now - timedelta(seconds=_BACKFILL_WINDOW_SECONDS)
    candles = _fetch_coinbase_candles(start, now)
    if not candles:
        return False

    seeded_samples = 0
    for candle_time, candle_open, candle_close in candles:
        for offset_seconds, price in ((0, candle_open), (20, candle_close), (40, candle_close)):
            sample_time = candle_time + timedelta(seconds=offset_seconds)
            if sample_time > now:
                continue
            _record_price_sample(price, as_of=sample_time)
            seeded_samples += 1

    return seeded_samples >= 15


def ensure_price_history_backfilled(now: Optional[datetime] = None) -> None:
    global _PRICE_HISTORY_BACKFILLED
    if _PRICE_HISTORY_BACKFILLED or _PRICE_HISTORY:
        _PRICE_HISTORY_BACKFILLED = True
        return

    as_of = now or datetime.now(timezone.utc)
    try:
        if _seed_price_history_from_trades(as_of):
            _PRICE_HISTORY_BACKFILLED = True
            return
    except requests.RequestException:
        pass

    try:
        if _seed_price_history_from_candles(as_of):
            _PRICE_HISTORY_BACKFILLED = True
            return
    except requests.RequestException:
        pass


def fetch_btc_spot_price(allow_cached_fallback: bool = True) -> float:
    global _LAST_SUCCESSFUL_PROVIDER_INDEX

    providers = _get_price_providers()
    provider_count = len(providers)
    last_error = None

    for offset in range(provider_count):
        provider_index = (_LAST_SUCCESSFUL_PROVIDER_INDEX + offset) % provider_count
        _, provider = providers[provider_index]
        try:
            price = provider()
            _LAST_SUCCESSFUL_PROVIDER_INDEX = provider_index
            _record_price_sample(price)
            return price
        except requests.RequestException as exc:
            last_error = exc

    if allow_cached_fallback:
        cached_price = get_latest_cached_price()
        if cached_price is not None:
            return cached_price

    if last_error is not None:
        raise last_error

    raise RuntimeError("No BTC price provider returned a price")


def _compute_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None

    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = prices[i] - prices[i - 1]
        if diff >= 0:
            gains.append(diff)
        else:
            losses.append(-diff)

    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _get_latest_price_at_or_before(cutoff: datetime) -> Optional[float]:
    for ts, price in reversed(_PRICE_HISTORY):
        if ts <= cutoff:
            return price
    return None


def _compute_velocity(now: datetime, price_now: float, seconds: int) -> Optional[float]:
    reference_price = _get_latest_price_at_or_before(now - timedelta(seconds=seconds))
    if reference_price is None:
        return None
    return price_now - reference_price


def _count_consecutive_flat_ticks(prices: List[float], epsilon: float = 1e-9) -> int:
    if len(prices) < 2:
        return 0
    flat_count = 0
    for idx in range(len(prices) - 1, 0, -1):
        if abs(prices[idx] - prices[idx - 1]) <= epsilon:
            flat_count += 1
            continue
        break
    return flat_count


def _get_market_window_reference_sample(
    window_start: datetime,
    max_lookback_seconds: int = _WINDOW_BASELINE_CARRY_FORWARD_SECONDS,
    max_lookahead_seconds: int = _WINDOW_BASELINE_LOOKAHEAD_SECONDS,
) -> Optional[Tuple[datetime, float]]:
    latest_before = next(
        (
            (ts, price)
            for ts, price in reversed(_PRICE_HISTORY)
            if ts <= window_start and (window_start - ts).total_seconds() <= max_lookback_seconds
        ),
        None,
    )
    if latest_before is not None:
        return latest_before

    earliest_after = next(
        (
            (ts, price)
            for ts, price in _PRICE_HISTORY
            if ts > window_start and (ts - window_start).total_seconds() <= max_lookahead_seconds
        ),
        None,
    )
    return earliest_after


def estimate_market_window_reference_price(
    window_start_ts: int,
    now: Optional[datetime] = None,
) -> Optional[float]:
    as_of = now or datetime.now(timezone.utc)
    ensure_price_history_backfilled(as_of)
    window_start = datetime.fromtimestamp(window_start_ts, tz=timezone.utc)
    reference_sample = _get_market_window_reference_sample(window_start)
    if reference_sample is None:
        return None
    return reference_sample[1]


def build_btc_features(window_start_ts: int) -> BtcFeatures:
    """
    Build BTC feature snapshot for the current 5-minute window.

    - Grabs a single fresh BTC spot price.
    - Uses a short rolling in-memory history for RSI/momentum/vol.
    - Approximates 'window open price' as the earliest price in the last ~N samples.
    """
    now = datetime.now(timezone.utc)
    ensure_price_history_backfilled(now)
    price_now = fetch_btc_spot_price()

    prices = [p[1] for p in _PRICE_HISTORY]
    window_start = datetime.fromtimestamp(window_start_ts, tz=timezone.utc)
    current_window_samples = [(ts, price) for ts, price in _PRICE_HISTORY if ts >= window_start]
    prior_window_sample = _get_market_window_reference_sample(window_start)
    effective_window_samples = list(current_window_samples)
    if prior_window_sample is not None and prior_window_sample[0] < window_start:
        effective_window_samples.insert(0, prior_window_sample)

    window_prices = [price for _, price in effective_window_samples]
    trailing_5m_cutoff = now - timedelta(seconds=300)
    trailing_5m_samples = [(ts, price) for ts, price in _PRICE_HISTORY if ts >= trailing_5m_cutoff]
    trailing_5m_prices = [price for _, price in trailing_5m_samples]
    one_minute_cutoff = now.timestamp() - 60
    one_minute_prices = [price for ts, price in _PRICE_HISTORY if ts.timestamp() >= one_minute_cutoff]

    # Carry the last pre-window sample forward so a new 5-minute period can use
    # the already-retained history immediately instead of waiting for an extra tick.
    window_open_price = (
        prior_window_sample[1]
        if prior_window_sample is not None
        else (window_prices[0] if window_prices else price_now)
    )

    trailing_5m_open_price = trailing_5m_prices[0] if trailing_5m_prices else price_now

    delta_pct = (price_now - window_open_price) / window_open_price if window_open_price else 0.0
    trailing_5m_delta_pct = (
        (price_now - trailing_5m_open_price) / trailing_5m_open_price
        if trailing_5m_open_price
        else 0.0
    )
    delta_from_previous_tick = price_now - prices[-2] if len(prices) >= 2 else None
    rsi = _compute_rsi(prices[-15:])
    momentum_1m = price_now - one_minute_prices[0] if len(one_minute_prices) >= 2 else None
    momentum_5m = price_now - trailing_5m_open_price if len(trailing_5m_prices) >= 2 else None
    velocity_15s = _compute_velocity(now, price_now, 15)
    velocity_30s = _compute_velocity(now, price_now, 30)
    volatility_5m = statistics.pstdev(trailing_5m_prices) if len(trailing_5m_prices) >= 2 else None
    consecutive_flat_ticks = _count_consecutive_flat_ticks(prices)

    return BtcFeatures(
        as_of=now,
        price_usd=price_now,
        window_open_price=window_open_price,
        trailing_5m_open_price=trailing_5m_open_price,
        delta_pct_from_window_open=delta_pct,
        delta_pct_from_trailing_5m_open=trailing_5m_delta_pct,
        delta_from_previous_tick=delta_from_previous_tick,
        rsi_14=rsi,
        momentum_1m=momentum_1m,
        momentum_5m=momentum_5m,
        velocity_15s=velocity_15s,
        velocity_30s=velocity_30s,
        volatility_5m=volatility_5m,
        consecutive_flat_ticks=consecutive_flat_ticks,
        retained_sample_count=len(prices),
        window_sample_count=len(window_prices),
        trailing_5m_sample_count=len(trailing_5m_prices),
    )


def get_feature_readiness(features: BtcFeatures) -> Tuple[bool, str]:
    reasons = []

    if features.rsi_14 is None:
        samples_needed = max(15 - features.retained_sample_count, 0)
        reasons.append(
            f"RSI warmup incomplete ({features.retained_sample_count}/15 samples"
            + (f", need {samples_needed} more" if samples_needed else "")
            + ")"
        )

    if features.momentum_5m is None or features.volatility_5m is None:
        window_needed = max(2 - features.trailing_5m_sample_count, 0)
        reasons.append(
            f"trailing 5-minute warmup incomplete "
            f"({features.trailing_5m_sample_count}/2 samples"
            + (f", need {window_needed} more" if window_needed else "")
            + ")"
        )

    if reasons:
        return False, "; ".join(reasons)

    return True, "ready"
