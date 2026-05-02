# custom/btc_agent/config.py

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional

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
    max_trade_usd: float = 5.0
    trade_shares_size: float = 5.0
    max_trades_per_period: int = 1
    max_automated_loss_trades: int = 0
    min_confidence: float = 0.7
    max_entry_price: float = 0.62
    max_spread: float = 0.06
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
        max_trade_usd=float(os.getenv("BTC_AGENT_MAX_TRADE_USD", "5")),
        trade_shares_size=max(float(os.getenv("BTC_AGENT_TRADE_SHARES_SIZE", "5")), 0.0),
        max_trades_per_period=max(int(os.getenv("BTC_AGENT_MAX_TRADES_PER_PERIOD", "1")), 1),
        max_automated_loss_trades=max(int(os.getenv("MAX_AUTOMATED_LOSS_TRADES", "0")), 0),
        min_confidence=float(
            os.getenv("CONFIDENCE", os.getenv("BTC_AGENT_MIN_CONFIDENCE", "0.7"))
        ),
        max_entry_price=float(os.getenv("BTC_AGENT_MAX_ENTRY_PRICE", "0.62")),
        max_spread=float(os.getenv("BTC_AGENT_MAX_SPREAD", "0.06")),
        market_slug_override=os.getenv("BTC_AGENT_MARKET_SLUG"),
    )
