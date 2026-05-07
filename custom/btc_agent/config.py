# custom/btc_agent/config.py

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional

# Base minimum confidence required for a directional trade before any
# time-remaining or trend-strength adjustments are applied.
# Example: raising this from 0.65 to 0.72 reduces trade count by forcing
# stronger raw LLM conviction; lowering it to 0.60 allows more marginal trades.
DEFAULT_MIN_CONFIDENCE = 0.65

# Fixed number of shares to submit for each paper/live trade.
# Example: increasing from 5 to 10 doubles position size and PnL variance;
# lowering to 3 reduces exposure per order.
SHARES_PER_TRADE = 5

# ADX level where the bot should still treat early-window conditions with caution.
# This is mainly a prompt/regime threshold for "Discovery Phase" behavior.
# Example: lowering from 15 to 10 makes the bot more willing to view weak trends
# as actionable; raising to 20 makes early-window trading more conservative.
DISCOVERY_ADX_CAUTION_THRESHOLD = 15.0

# ADX level that marks a trend as strong enough to relax the normal confidence floor.
# Example: lowering from 30 to 25 will allow more strong-trend trades to pass with
# slightly lower confidence; raising to 35 means only very strong trends get that benefit.
TREND_PRIORITY_ADX_THRESHOLD = 30.0

# Relaxed minimum confidence used when ADX is above TREND_PRIORITY_ADX_THRESHOLD
# and there is still meaningful time left in the period.
# Example: lowering from 0.62 to 0.58 increases order rate in strong trends;
# raising to 0.66 keeps the bot stricter even when momentum is obvious.
TREND_RELAXED_MIN_CONFIDENCE = 0.62

# Harder minimum confidence floor used in the final minute of the period.
# Example: raising from 0.75 to 0.80 blocks more late-window trades; lowering
# to 0.70 makes the bot more willing to take last-minute entries.
FINAL_WINDOW_MIN_CONFIDENCE = 0.75

# Minimum required execution edge above market implied probability for a trade
# to pass, except for special very-high-confidence override cases.
# Example: raising from 0.02 to 0.05 makes the bot demand more pricing edge;
# lowering to 0.00 allows trades closer to fair value.
MIN_EXECUTION_EDGE = 0.02

# Distance in USD beyond the strike that the chosen side must already be winning
# by before the in-the-money confidence boost can apply.
# Example: lowering from 20 to 10 lets smaller leads earn a confidence boost;
# raising to 30 means only clearly in-the-money trades get that boost.
ITM_CONFIDENCE_BOOST_USD = 20.0

# Minimum market win chance required before the in-the-money confidence boost
# is allowed to apply.
# Example: lowering from 0.60 to 0.55 makes the boost trigger more often;
# raising to 0.70 restricts the boost to stronger market consensus cases.
ITM_CONFIDENCE_BOOST_MARKET_WIN_CHANCE = 0.60

# Amount added to decision confidence when the in-the-money boost conditions are met.
# Example: raising from 0.10 to 0.15 helps more trades clear confidence/edge gates;
# lowering to 0.05 makes the boost more modest.
ITM_CONFIDENCE_BOOST_AMOUNT = 0.10

# Low market-win-chance threshold used by the prompt/execution guardrails to avoid
# betting on extremely unlikely reversals.
# Example: lowering from 0.15 to 0.10 allows more contrarian entries; raising
# to 0.20 blocks more low-probability fade attempts.
MARKET_WIN_CHANCE_VETO_THRESHOLD = 0.15

# Time-remaining boundary for the market-win-chance veto window.
# Example: lowering from 120 to 90 shortens the veto period and allows more trades
# earlier in the window; raising to 180 keeps the reversal veto active longer.
MARKET_WIN_CHANCE_VETO_END_SECONDS = 120

# Base RSI ceiling for blocking UP trades in overbought conditions.
# This base threshold applies when trend strength is not extreme.
# Example: lowering from 70 to 65 blocks more UP trades on stretched moves;
# raising to 75 allows more breakout continuation attempts.
UP_RSI_VETO_BASE_THRESHOLD = 70.0

# Higher RSI ceiling for blocking UP trades when ADX confirms a strong trend.
# Example: lowering from 85 to 80 makes the bot stop chasing strong-up moves sooner;
# raising to 90 allows more late continuation trades before vetoing them.
UP_RSI_VETO_TREND_THRESHOLD = 85.0

# ADX threshold that switches the UP RSI veto from the base threshold to the
# stronger-trend threshold above.
# Example: lowering from 30 to 25 makes the trend-aware RSI ceiling engage earlier;
# raising to 35 means only stronger trends get the looser RSI allowance.
UP_RSI_VETO_ADX_THRESHOLD = 30.0

# RSI floor for blocking DOWN trades in oversold conditions.
# Example: lowering from 30 to 25 allows more downside continuation trades;
# raising to 35 vetoes more "falling knife" short entries.
DOWN_RSI_VETO_THRESHOLD = 30.0

# Divisor used in the required-velocity sanity check.
# The bot rejects trades when required_velocity_to_win exceeds volatility_5m / divisor.
# Example: raising from 15 to 20 makes the check stricter and blocks more
# mathematically difficult trades; lowering to 10 makes it more permissive.
REQUIRED_VELOCITY_DIVISOR = 15.0

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(REPO_ROOT, ".env"))


def _parse_rpc_urls() -> list[str]:
    raw = os.getenv("POLYGON_RPC_URLS", "").strip()
    if raw:
        urls = [url.strip() for url in raw.split(",") if url.strip()]
        if urls:
            return urls

    primary = os.getenv("POLYGON_RPC_URL", "https://polygon.drpc.org").strip()

    fallbacks = [
        "https://polygon.publicnode.com",
        "https://tenderly.rpc.polygon.community",
    ]

    urls = [primary]
    for url in fallbacks:
        if url not in urls:
            urls.append(url)
    return urls


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

@dataclass
class LlmConfig:
    engine: str
    api_key: str
    model: str
    api_connection_timeout_seconds: float = 10.0
    api_connection_retry_timer_seconds: float = 2.0
    api_connection_retry_attempts: int = 3

@dataclass
class TradingConfig:
    paper_trading: bool = True
    debug: bool = False
    debug_price_to_beat: bool = False
    llm_connection_debug: bool = False
    minimum_wallet_balance: float = 0.0
    live_fee_rate_bps: int = 1000
    live_min_order_usd: float = 1.0
    use_recommended_limit: bool = True
    disable_liquidity_filter: bool = False
    shares_per_trade: float = SHARES_PER_TRADE
    max_trades_per_period: int = 1
    max_periods_per_run: int = 0
    max_automated_loss_trades: int = 0
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    max_entry_price: float = 0.62
    max_spread: float = 0.06
    discovery_adx_caution_threshold: float = DISCOVERY_ADX_CAUTION_THRESHOLD
    trend_priority_adx_threshold: float = TREND_PRIORITY_ADX_THRESHOLD
    trend_relaxed_min_confidence: float = TREND_RELAXED_MIN_CONFIDENCE
    final_window_min_confidence: float = FINAL_WINDOW_MIN_CONFIDENCE
    min_execution_edge: float = MIN_EXECUTION_EDGE
    itm_confidence_boost_usd: float = ITM_CONFIDENCE_BOOST_USD
    itm_confidence_boost_market_win_chance: float = ITM_CONFIDENCE_BOOST_MARKET_WIN_CHANCE
    itm_confidence_boost_amount: float = ITM_CONFIDENCE_BOOST_AMOUNT
    market_win_chance_veto_threshold: float = MARKET_WIN_CHANCE_VETO_THRESHOLD
    market_win_chance_veto_end_seconds: int = MARKET_WIN_CHANCE_VETO_END_SECONDS
    up_rsi_veto_base_threshold: float = UP_RSI_VETO_BASE_THRESHOLD
    up_rsi_veto_trend_threshold: float = UP_RSI_VETO_TREND_THRESHOLD
    up_rsi_veto_adx_threshold: float = UP_RSI_VETO_ADX_THRESHOLD
    down_rsi_veto_threshold: float = DOWN_RSI_VETO_THRESHOLD
    required_velocity_divisor: float = REQUIRED_VELOCITY_DIVISOR
    market_slug_override: Optional[str] = None

@dataclass
class PolymarketConfig:
    private_key: str
    proxy_address: Optional[str]
    gamma_api: str = "https://gamma-api.polymarket.com"
    data_api: str = "https://data-api.polymarket.com"
    clob_api: str = "https://clob.polymarket.com"
    polygon_rpc: str = _parse_rpc_urls()[0]
    polygon_rpc_urls: list[str] = None
    chain_id: int = 137

def get_llm_config() -> LlmConfig:
    raw_engine = os.getenv("AI_ENGINE", "OPENAI").strip().lower()
    raw_timeout = os.getenv("API_CONNECTION_TIMEOUT")
    if raw_timeout is None:
        raw_timeout = os.getenv("API_CONNECTION_TMEOUT", "10")
    api_connection_timeout_seconds = max(float(raw_timeout), 0.1)
    api_connection_retry_timer_seconds = max(
        float(os.getenv("API_CONNECTION_RETRY_TIMER", "2.0")),
        0.0,
    )
    api_connection_retry_attempts = max(
        int(os.getenv("API_CONNECTION_RETRY_ATTEMPTS", "3")),
        1,
    )

    if raw_engine == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
        return LlmConfig(
            engine="openai",
            api_key=api_key,
            model=model,
            api_connection_timeout_seconds=api_connection_timeout_seconds,
            api_connection_retry_timer_seconds=api_connection_retry_timer_seconds,
            api_connection_retry_attempts=api_connection_retry_attempts,
        )

    if raw_engine in {"gemini", "google"}:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
        return LlmConfig(
            engine="gemini",
            api_key=api_key,
            model=model,
            api_connection_timeout_seconds=api_connection_timeout_seconds,
            api_connection_retry_timer_seconds=api_connection_retry_timer_seconds,
            api_connection_retry_attempts=api_connection_retry_attempts,
        )

    raise RuntimeError("AI_ENGINE must be one of: OPENAI, GEMINI")

def get_polymarket_config() -> PolymarketConfig:
    pk = os.getenv("POLYGON_WALLET_PRIVATE_KEY")
    if not pk:
        raise RuntimeError("POLYGON_WALLET_PRIVATE_KEY is not set in .env")
    proxy = os.getenv("POLYMKT_PROXY_ADDRESS")
    rpc_urls = _parse_rpc_urls()
    return PolymarketConfig(
        private_key=pk,
        proxy_address=proxy,
        polygon_rpc=rpc_urls[0],
        polygon_rpc_urls=rpc_urls,
    )

def get_trading_config() -> TradingConfig:
    return TradingConfig(
        paper_trading=_parse_bool_env("USE_PAPER_TRADES", True),
        debug=_parse_bool_env("BTC_AGENT_DEBUG", False),
        debug_price_to_beat=_parse_bool_env("DEBUG_PRICE_TO_BEAT", False),
        llm_connection_debug=_parse_bool_env("LLM_CONNECTION_DEBUG", False),
        minimum_wallet_balance=float(os.getenv("MINIMUM_WALLET_BALANCE", "0")),
        live_fee_rate_bps=int(os.getenv("BTC_AGENT_LIVE_FEE_RATE_BPS", "1000")),
        live_min_order_usd=float(os.getenv("BTC_AGENT_LIVE_MIN_ORDER_USD", "1")),
        use_recommended_limit=_parse_bool_env("USE_RECOMMENDED_LIMIT", True),
        disable_liquidity_filter=_parse_bool_env("DISABLE_LIQUIDITY_FILTER", False),
        shares_per_trade=float(SHARES_PER_TRADE),
        max_trades_per_period=max(int(os.getenv("BTC_AGENT_MAX_TRADES_PER_PERIOD", "1")), 1),
        max_periods_per_run=max(int(os.getenv("MAX_PERIODS_PER_RUN", "0")), 0),
        max_automated_loss_trades=max(int(os.getenv("MAX_AUTOMATED_LOSS_TRADES", "0")), 0),
        min_confidence=float(os.getenv("BTC_AGENT_MIN_CONFIDENCE", str(DEFAULT_MIN_CONFIDENCE))),
        max_entry_price=float(os.getenv("BTC_AGENT_MAX_ENTRY_PRICE", "0.62")),
        max_spread=float(os.getenv("BTC_AGENT_MAX_SPREAD", "0.06")),
        discovery_adx_caution_threshold=float(
            os.getenv("BTC_AGENT_DISCOVERY_ADX_CAUTION_THRESHOLD", str(DISCOVERY_ADX_CAUTION_THRESHOLD))
        ),
        trend_priority_adx_threshold=float(
            os.getenv("BTC_AGENT_TREND_PRIORITY_ADX_THRESHOLD", str(TREND_PRIORITY_ADX_THRESHOLD))
        ),
        trend_relaxed_min_confidence=float(
            os.getenv("BTC_AGENT_TREND_RELAXED_MIN_CONFIDENCE", str(TREND_RELAXED_MIN_CONFIDENCE))
        ),
        final_window_min_confidence=float(
            os.getenv("BTC_AGENT_FINAL_WINDOW_MIN_CONFIDENCE", str(FINAL_WINDOW_MIN_CONFIDENCE))
        ),
        min_execution_edge=float(
            os.getenv("BTC_AGENT_MIN_EXECUTION_EDGE", str(MIN_EXECUTION_EDGE))
        ),
        itm_confidence_boost_usd=float(
            os.getenv("BTC_AGENT_ITM_CONFIDENCE_BOOST_USD", str(ITM_CONFIDENCE_BOOST_USD))
        ),
        itm_confidence_boost_market_win_chance=float(
            os.getenv(
                "BTC_AGENT_ITM_CONFIDENCE_BOOST_MARKET_WIN_CHANCE",
                str(ITM_CONFIDENCE_BOOST_MARKET_WIN_CHANCE),
            )
        ),
        itm_confidence_boost_amount=float(
            os.getenv("BTC_AGENT_ITM_CONFIDENCE_BOOST_AMOUNT", str(ITM_CONFIDENCE_BOOST_AMOUNT))
        ),
        market_win_chance_veto_threshold=float(
            os.getenv("BTC_AGENT_MARKET_WIN_CHANCE_VETO_THRESHOLD", str(MARKET_WIN_CHANCE_VETO_THRESHOLD))
        ),
        market_win_chance_veto_end_seconds=max(
            int(
                os.getenv(
                    "BTC_AGENT_MARKET_WIN_CHANCE_VETO_END_SECONDS",
                    str(MARKET_WIN_CHANCE_VETO_END_SECONDS),
                )
            ),
            0,
        ),
        up_rsi_veto_base_threshold=float(
            os.getenv("BTC_AGENT_UP_RSI_VETO_BASE_THRESHOLD", str(UP_RSI_VETO_BASE_THRESHOLD))
        ),
        up_rsi_veto_trend_threshold=float(
            os.getenv("BTC_AGENT_UP_RSI_VETO_TREND_THRESHOLD", str(UP_RSI_VETO_TREND_THRESHOLD))
        ),
        up_rsi_veto_adx_threshold=float(
            os.getenv("BTC_AGENT_UP_RSI_VETO_ADX_THRESHOLD", str(UP_RSI_VETO_ADX_THRESHOLD))
        ),
        down_rsi_veto_threshold=float(
            os.getenv("BTC_AGENT_DOWN_RSI_VETO_THRESHOLD", str(DOWN_RSI_VETO_THRESHOLD))
        ),
        required_velocity_divisor=float(
            os.getenv("BTC_AGENT_REQUIRED_VELOCITY_DIVISOR", str(REQUIRED_VELOCITY_DIVISOR))
        ),
        market_slug_override=os.getenv("BTC_AGENT_MARKET_SLUG"),
    )
