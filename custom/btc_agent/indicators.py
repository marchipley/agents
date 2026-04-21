# custom/btc_agent/indicators.py

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Tuple
import statistics

import requests

# Simple in-memory history: list of (timestamp, price)
_PRICE_HISTORY: List[Tuple[datetime, float]] = []
_LAST_SUCCESSFUL_PROVIDER_INDEX = 0


@dataclass
class BtcFeatures:
    as_of: datetime
    price_usd: float
    window_open_price: float
    delta_pct_from_window_open: float
    rsi_14: Optional[float]
    momentum_5m: Optional[float]
    volatility_5m: Optional[float]


def _fetch_spot_price_from_coingecko() -> float:
    resp = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": "bitcoin", "vs_currencies": "usd"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return float(data["bitcoin"]["usd"])


def _fetch_spot_price_from_coinbase() -> float:
    resp = requests.get(
        "https://api.coinbase.com/v2/prices/BTC-USD/spot",
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return float(data["data"]["amount"])


def _get_price_providers():
    return [
        ("CoinGecko", _fetch_spot_price_from_coingecko),
        ("Coinbase", _fetch_spot_price_from_coinbase),
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


def build_btc_features(window_start_ts: int) -> BtcFeatures:
    """
    Build BTC feature snapshot for the current 5-minute window.

    - Grabs a single fresh BTC spot price.
    - Uses a short rolling in-memory history for RSI/momentum/vol.
    - Approximates 'window open price' as the earliest price in the last ~N samples.
    """
    now = datetime.now(timezone.utc)
    price_now = fetch_btc_spot_price()

    prices = [p[1] for p in _PRICE_HISTORY]

    # Approximate window open price as the earliest price we still have
    window_open_price = prices[0] if prices else price_now

    delta_pct = (price_now - window_open_price) / window_open_price if window_open_price else 0.0
    rsi = _compute_rsi(prices[-15:])
    momentum_5m = price_now - window_open_price if len(prices) >= 2 else None
    volatility_5m = statistics.pstdev(prices[-15:]) if len(prices) >= 2 else None

    return BtcFeatures(
        as_of=now,
        price_usd=price_now,
        window_open_price=window_open_price,
        delta_pct_from_window_open=delta_pct,
        rsi_14=rsi,
        momentum_5m=momentum_5m,
        volatility_5m=volatility_5m,
    )
