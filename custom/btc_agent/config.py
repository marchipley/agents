# custom/btc_agent/config.py

import os
import sys
import importlib
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional

# Base minimum confidence required for a directional trade before any
# time-remaining or trend-strength adjustments are applied.
# Example: raising this from 0.65 to 0.72 reduces trade count by forcing
# stronger raw LLM conviction; lowering it to 0.60 allows more marginal trades.
DEFAULT_MIN_CONFIDENCE = 0.65

# Very low confidence floor used in the early Discovery Phase to allow
# mathematically favorable early entries before the market fully trends.
# Example: raising from 0.10 to 0.25 makes early entries rarer; lowering it
# further makes the bot more willing to probe early-window setups.
DISCOVERY_MIN_CONFIDENCE = 0.85

# Fixed number of shares to submit for each paper/live trade.
# Example: increasing from 5 to 10 doubles position size and PnL variance;
# lowering to 3 reduces exposure per order.
SHARES_PER_TRADE = 5

# Maximum spread allowed by configuration for a trade candidate.
# Example: raising from 0.06 to 0.10 allows participation in wider, more
# volatile books; lowering it makes the bot stricter about entry quality.
DEFAULT_MAX_SPREAD = 0.20

# ADX level where the bot should still treat early-window conditions with caution.
# This is mainly a prompt/regime threshold for "Discovery Phase" behavior.
# Example: lowering from 15 to 10 makes the bot more willing to view weak trends
# as actionable; raising to 20 makes early-window trading more conservative.
DISCOVERY_ADX_CAUTION_THRESHOLD = 20.0

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
FINAL_WINDOW_MIN_CONFIDENCE = 0.70

# Minimum required execution edge above market implied probability for a trade
# to pass, except for special very-high-confidence override cases.
# Example: raising from 0.02 to 0.05 makes the bot demand more pricing edge;
# lowering to 0.00 allows trades closer to fair value.
MIN_EXECUTION_EDGE = 0.02

# Legacy fixed-USD fallback for the in-the-money confidence boost. The active
# logic now prefers an ATR-based threshold when ATR is available.
# Example: lowering from 20 to 10 lets smaller leads earn a fallback boost;
# raising to 30 means only clearly in-the-money trades get that fallback.
ITM_CONFIDENCE_BOOST_USD = 20.0

# ATR multiplier used to decide when a side is winning by enough to deserve
# the in-the-money confidence boost.
# Example: lowering from 0.50 to 0.25 makes the boost trigger earlier; raising
# to 0.75 requires a larger cushion over the strike before boosting confidence.
ITM_CONFIDENCE_BOOST_ATR_MULTIPLIER = 0.50

# Minimum market win chance required before the in-the-money confidence boost
# is allowed to apply.
# Example: lowering from 0.60 to 0.55 makes the boost trigger more often;
# raising to 0.70 restricts the boost to stronger market consensus cases.
ITM_CONFIDENCE_BOOST_MARKET_WIN_CHANCE = 0.60

# Amount added to decision confidence when the in-the-money boost conditions are met.
# Example: raising from 0.15 to 0.20 helps more trades clear confidence/edge gates;
# lowering to 0.10 makes the boost more modest.
ITM_CONFIDENCE_BOOST_AMOUNT = 0.15

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

# When RSI(9) - RSI(14) exceeds this value in a strong trend, the normal UP RSI
# veto can be suspended because accelerating RSI is being treated as breakout
# continuation instead of exhaustion.
# Example: lowering from 5 to 3 makes the suspension happen more often; raising
# to 7 requires a sharper RSI acceleration before allowing parabolic continuation.
PARABOLIC_RSI_SPEED_DIVERGENCE_THRESHOLD = 5.0

# ADX threshold required before the parabolic RSI suspension is allowed.
# Example: lowering from 35 to 30 allows more breakouts to bypass the RSI veto;
# raising to 40 reserves the suspension for only the strongest trends.
PARABOLIC_RSI_SUSPEND_ADX_THRESHOLD = 35.0

# RSI floor for blocking DOWN trades in oversold conditions.
# Example: lowering from 30 to 25 allows more downside continuation trades;
# raising to 35 vetoes more "falling knife" short entries.
DOWN_RSI_VETO_THRESHOLD = 30.0

# Stricter RSI ceiling for new trades placed early in the window when there is
# still enough time for mean reversion noise to punish an overextended entry.
# Example: lowering from 65 to 60 blocks more early momentum chases; raising it
# to 70 allows more aggressive early trend-following.
MAX_EARLY_WINDOW_RSI = 65.0

# Divisor used in the required-velocity sanity check.
# The bot rejects trades when required_velocity_to_win exceeds volatility_5m / divisor.
# Example: raising from 15 to 20 makes the check stricter and blocks more
# mathematically difficult trades; lowering to 10 makes it more permissive.
REQUIRED_VELOCITY_DIVISOR = 7.5

# Early-window volatility-to-strike-gap safety multiplier used to block
# trades where BTC is too close to the strike relative to current 5m noise.
# Example: raising from 0.4 to 0.5 requires a larger lead early; lowering to
# 0.3 allows more early entries near the strike.
DYNAMIC_STRIKE_BUFFER_MULTIPLIER = 0.4

# When order-book imbalance pressure is strongly positive for the chosen side,
# shift the reference price toward the ask to favor faster fills.
# Example: lowering this from 0.50 to 0.35 makes the bot react sooner to
# imbalance; raising it makes the shift rarer.
IMBALANCE_PRICING_THRESHOLD = 0.50

# Stronger imbalance threshold that justifies a 2-tick shift instead of 1 tick.
# Example: lowering from 0.75 to 0.65 makes aggressive repricing more common.
IMBALANCE_PRICING_STRONG_THRESHOLD = 0.75

# Early-window safety buffer multiplier for strike-distance filtering.
# Example: raising from 0.50 to 0.65 makes the bot wait for a larger lead early
# in the period; lowering it allows more early near-strike entries.
EARLY_WINDOW_BUFFER_MULTIPLIER = 0.50

# Late-window safety buffer multiplier for strike-distance filtering.
# Example: lowering from 0.15 to 0.10 allows more late near-strike trades;
# raising it makes the bot stay more conservative closer to expiry.
LATE_WINDOW_BUFFER_MULTIPLIER = 0.15

# Realized slippage guard in basis points. If the confirmed fill exceeds this
# slippage relative to the quoted entry price, trigger a cooldown.
# Example: lowering from 500 to 300 reacts faster to unstable books; raising it
# tolerates more slippage before cooling down.
SLIPPAGE_COOLDOWN_THRESHOLD_BPS = 500.0

# Cooldown duration, in seconds, after a trade fills with extreme slippage.
# Example: raising from 300 to 600 pauses the strategy longer after a bad fill.
SLIPPAGE_COOLDOWN_SECONDS = 300

# The total number of completed losing trades allowed in a single run before the
# bot stops. This is a repo-visible non-secret runtime cap.
MAX_LOSSES_PER_RUN = 4

# Number of times to poll Polymarket order status right after live submission
# before deciding the order is still unfilled.
# Example: raising from 4 to 8 waits longer for exchange confirmation; lowering
# to 2 makes the bot classify unfilled orders faster but with less certainty.
LIVE_ORDER_STATUS_POLL_ATTEMPTS = 4

# Delay between live order-status polls after submission.
# Example: raising from 0.75s to 1.5s gives more time for matching but slows
# the loop; lowering to 0.25s checks faster but can miss slower confirmations.
LIVE_ORDER_STATUS_POLL_INTERVAL_SECONDS = 0.75

# Enabling tis will output additional debug info in the output
BTC_AGENT_DEBUG=False

# Enabling this will disable warmup if BTC_AGENT_DEBUG is enabled
BTC_AGENT_DEBUG_WARMUP=False

# When enabled, the program will disable the geolocation check and send a small test prompt to the LLM to check connection
# then do a test connection to google if it fails so we can see if there is any basic connection
# issues. If that succeeds, it will then send a full size system and user prompt to the LLM and confirm 
# a response then the script will terminate and advise the user that the script was terminated due to LLM_CONNECTION_DEBUG=True
# and to change it in order to run all other script features.
# This will allow us to troubleshoot issues we are having the the reponses on the VPN and proxy

LLM_CONNECTION_DEBUG=False

BTC_AGENT_LOOP_INTERVAL=5
USE_PAPER_TRADES=False
BTC_AGENT_MAX_PRICE=2

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(REPO_ROOT, ".env"))


def reload_runtime_config_module():
    return importlib.reload(sys.modules[__name__])


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
    debug_warmup: bool = True
    llm_connection_debug: bool = False
    loop_interval_seconds: int = 30
    minimum_wallet_balance: float = 0.0
    live_fee_rate_bps: int = 1000
    live_min_order_usd: float = 1.0
    use_recommended_limit: bool = True
    disable_liquidity_filter: bool = False
    shares_per_trade: float = SHARES_PER_TRADE
    max_trades_per_period: int = 1
    max_losses_per_run: int = MAX_LOSSES_PER_RUN
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    discovery_min_confidence: float = DISCOVERY_MIN_CONFIDENCE
    max_entry_price: float = 0.62
    max_spread: float = DEFAULT_MAX_SPREAD
    discovery_adx_caution_threshold: float = DISCOVERY_ADX_CAUTION_THRESHOLD
    trend_priority_adx_threshold: float = TREND_PRIORITY_ADX_THRESHOLD
    trend_relaxed_min_confidence: float = TREND_RELAXED_MIN_CONFIDENCE
    final_window_min_confidence: float = FINAL_WINDOW_MIN_CONFIDENCE
    min_execution_edge: float = MIN_EXECUTION_EDGE
    itm_confidence_boost_usd: float = ITM_CONFIDENCE_BOOST_USD
    itm_confidence_boost_atr_multiplier: float = ITM_CONFIDENCE_BOOST_ATR_MULTIPLIER
    itm_confidence_boost_market_win_chance: float = ITM_CONFIDENCE_BOOST_MARKET_WIN_CHANCE
    itm_confidence_boost_amount: float = ITM_CONFIDENCE_BOOST_AMOUNT
    market_win_chance_veto_threshold: float = MARKET_WIN_CHANCE_VETO_THRESHOLD
    market_win_chance_veto_end_seconds: int = MARKET_WIN_CHANCE_VETO_END_SECONDS
    up_rsi_veto_base_threshold: float = UP_RSI_VETO_BASE_THRESHOLD
    up_rsi_veto_trend_threshold: float = UP_RSI_VETO_TREND_THRESHOLD
    up_rsi_veto_adx_threshold: float = UP_RSI_VETO_ADX_THRESHOLD
    parabolic_rsi_speed_divergence_threshold: float = PARABOLIC_RSI_SPEED_DIVERGENCE_THRESHOLD
    parabolic_rsi_suspend_adx_threshold: float = PARABOLIC_RSI_SUSPEND_ADX_THRESHOLD
    down_rsi_veto_threshold: float = DOWN_RSI_VETO_THRESHOLD
    max_early_window_rsi: float = MAX_EARLY_WINDOW_RSI
    required_velocity_divisor: float = REQUIRED_VELOCITY_DIVISOR
    dynamic_strike_buffer_multiplier: float = DYNAMIC_STRIKE_BUFFER_MULTIPLIER
    imbalance_pricing_threshold: float = IMBALANCE_PRICING_THRESHOLD
    imbalance_pricing_strong_threshold: float = IMBALANCE_PRICING_STRONG_THRESHOLD
    early_window_buffer_multiplier: float = EARLY_WINDOW_BUFFER_MULTIPLIER
    late_window_buffer_multiplier: float = LATE_WINDOW_BUFFER_MULTIPLIER
    slippage_cooldown_threshold_bps: float = SLIPPAGE_COOLDOWN_THRESHOLD_BPS
    slippage_cooldown_seconds: int = SLIPPAGE_COOLDOWN_SECONDS
    live_order_status_poll_attempts: int = LIVE_ORDER_STATUS_POLL_ATTEMPTS
    live_order_status_poll_interval_seconds: float = LIVE_ORDER_STATUS_POLL_INTERVAL_SECONDS
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
        paper_trading=bool(USE_PAPER_TRADES),
        debug=bool(BTC_AGENT_DEBUG),
        debug_warmup=bool(BTC_AGENT_DEBUG_WARMUP),
        llm_connection_debug=bool(LLM_CONNECTION_DEBUG),
        loop_interval_seconds=max(int(BTC_AGENT_LOOP_INTERVAL), 1),
        minimum_wallet_balance=float(os.getenv("MINIMUM_WALLET_BALANCE", "0")),
        live_fee_rate_bps=int(os.getenv("BTC_AGENT_LIVE_FEE_RATE_BPS", "1000")),
        live_min_order_usd=float(os.getenv("BTC_AGENT_LIVE_MIN_ORDER_USD", "1")),
        use_recommended_limit=_parse_bool_env("USE_RECOMMENDED_LIMIT", True),
        disable_liquidity_filter=_parse_bool_env("DISABLE_LIQUIDITY_FILTER", False),
        shares_per_trade=float(SHARES_PER_TRADE),
        max_trades_per_period=max(int(os.getenv("BTC_AGENT_MAX_TRADES_PER_PERIOD", "1")), 1),
        max_losses_per_run=max(int(MAX_LOSSES_PER_RUN), 0),
        min_confidence=float(os.getenv("BTC_AGENT_MIN_CONFIDENCE", str(DEFAULT_MIN_CONFIDENCE))),
        discovery_min_confidence=float(
            os.getenv("BTC_AGENT_DISCOVERY_MIN_CONFIDENCE", str(DISCOVERY_MIN_CONFIDENCE))
        ),
        max_entry_price=float(BTC_AGENT_MAX_PRICE),
        max_spread=float(os.getenv("BTC_AGENT_MAX_SPREAD", str(DEFAULT_MAX_SPREAD))),
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
        itm_confidence_boost_atr_multiplier=float(
            os.getenv(
                "BTC_AGENT_ITM_CONFIDENCE_BOOST_ATR_MULTIPLIER",
                str(ITM_CONFIDENCE_BOOST_ATR_MULTIPLIER),
            )
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
        parabolic_rsi_speed_divergence_threshold=float(
            os.getenv(
                "BTC_AGENT_PARABOLIC_RSI_SPEED_DIVERGENCE_THRESHOLD",
                str(PARABOLIC_RSI_SPEED_DIVERGENCE_THRESHOLD),
            )
        ),
        parabolic_rsi_suspend_adx_threshold=float(
            os.getenv(
                "BTC_AGENT_PARABOLIC_RSI_SUSPEND_ADX_THRESHOLD",
                str(PARABOLIC_RSI_SUSPEND_ADX_THRESHOLD),
            )
        ),
        down_rsi_veto_threshold=float(
            os.getenv("BTC_AGENT_DOWN_RSI_VETO_THRESHOLD", str(DOWN_RSI_VETO_THRESHOLD))
        ),
        max_early_window_rsi=float(
            os.getenv("BTC_AGENT_MAX_EARLY_WINDOW_RSI", str(MAX_EARLY_WINDOW_RSI))
        ),
        required_velocity_divisor=float(
            os.getenv("BTC_AGENT_REQUIRED_VELOCITY_DIVISOR", str(REQUIRED_VELOCITY_DIVISOR))
        ),
        dynamic_strike_buffer_multiplier=float(
            os.getenv(
                "BTC_AGENT_DYNAMIC_STRIKE_BUFFER_MULTIPLIER",
                str(DYNAMIC_STRIKE_BUFFER_MULTIPLIER),
            )
        ),
        imbalance_pricing_threshold=float(
            os.getenv("BTC_AGENT_IMBALANCE_PRICING_THRESHOLD", str(IMBALANCE_PRICING_THRESHOLD))
        ),
        imbalance_pricing_strong_threshold=float(
            os.getenv(
                "BTC_AGENT_IMBALANCE_PRICING_STRONG_THRESHOLD",
                str(IMBALANCE_PRICING_STRONG_THRESHOLD),
            )
        ),
        early_window_buffer_multiplier=float(
            os.getenv(
                "BTC_AGENT_EARLY_WINDOW_BUFFER_MULTIPLIER",
                str(EARLY_WINDOW_BUFFER_MULTIPLIER),
            )
        ),
        late_window_buffer_multiplier=float(
            os.getenv(
                "BTC_AGENT_LATE_WINDOW_BUFFER_MULTIPLIER",
                str(LATE_WINDOW_BUFFER_MULTIPLIER),
            )
        ),
        slippage_cooldown_threshold_bps=float(
            os.getenv(
                "BTC_AGENT_SLIPPAGE_COOLDOWN_THRESHOLD_BPS",
                str(SLIPPAGE_COOLDOWN_THRESHOLD_BPS),
            )
        ),
        slippage_cooldown_seconds=max(
            int(os.getenv("BTC_AGENT_SLIPPAGE_COOLDOWN_SECONDS", str(SLIPPAGE_COOLDOWN_SECONDS))),
            0,
        ),
        live_order_status_poll_attempts=max(
            int(
                os.getenv(
                    "BTC_AGENT_LIVE_ORDER_STATUS_POLL_ATTEMPTS",
                    str(LIVE_ORDER_STATUS_POLL_ATTEMPTS),
                )
            ),
            1,
        ),
        live_order_status_poll_interval_seconds=max(
            float(
                os.getenv(
                    "BTC_AGENT_LIVE_ORDER_STATUS_POLL_INTERVAL_SECONDS",
                    str(LIVE_ORDER_STATUS_POLL_INTERVAL_SECONDS),
                )
            ),
            0.0,
        ),
        market_slug_override=os.getenv("BTC_AGENT_MARKET_SLUG"),
    )
