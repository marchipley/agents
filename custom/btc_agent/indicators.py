# custom/btc_agent/indicators.py

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Tuple
import statistics

import requests

# Simple in-memory history: list of (timestamp, price)
_PRICE_HISTORY: List[Tuple[datetime, float]] = []


@dataclass
class BtcFeatures:
    as_of: datetime
    price_usd: float
    window_open_price: float
    delta_pct_from_window_open: float
    rsi_14: Optional[float]
    momentum_5m: Optional[float]
    volatility_5m: Optional[float]


def _fetch_spot_price() -> float:
    """
    Fetch the current BTC/USD price from a public API.

    For now, use CoinGecko's simple endpoint once per run.
    """
    resp = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": "bitcoin", "vs_currencies": "usd"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return float(data["bitcoin"]["usd"])


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
    global _PRICE_HISTORY

    now = datetime.now(timezone.utc)
    price_now = _fetch_spot_price()

    # Append to history and keep only the last N samples (e.g., last 60)
    _PRICE_HISTORY.append((now, price_now))
    if len(_PRICE_HISTORY) > 60:
        _PRICE_HISTORY = _PRICE_HISTORY[-60:]

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
