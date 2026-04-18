# custom/btc_agent/config.py

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(REPO_ROOT, ".env"))

@dataclass
class OpenAIConfig:
    api_key: str
    model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

@dataclass
class TradingConfig:
    paper_trading: bool = True
    max_trade_usd: float = float(os.getenv("BTC_AGENT_MAX_TRADE_USD", "5"))
    min_confidence: float = float(os.getenv("BTC_AGENT_MIN_CONFIDENCE", "0.7"))
    max_entry_price: float = float(os.getenv("BTC_AGENT_MAX_ENTRY_PRICE", "0.62"))
    max_spread: float = float(os.getenv("BTC_AGENT_MAX_SPREAD", "0.06"))
    market_slug_override: Optional[str] = os.getenv("BTC_AGENT_MARKET_SLUG")

@dataclass
class PolymarketConfig:
    private_key: str
    proxy_address: Optional[str]
    gamma_api: str = "https://gamma-api.polymarket.com"
    data_api: str = "https://data-api.polymarket.com"
    clob_api: str = "https://clob.polymarket.com"
    chain_id: int = 137

def get_openai_config() -> OpenAIConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")
    return OpenAIConfig(api_key=api_key)

def get_polymarket_config() -> PolymarketConfig:
    pk = os.getenv("POLYGON_WALLET_PRIVATE_KEY")
    if not pk:
        raise RuntimeError("POLYGON_WALLET_PRIVATE_KEY is not set in .env")
    proxy = os.getenv("POLYMKT_PROXY_ADDRESS")
    return PolymarketConfig(private_key=pk, proxy_address=proxy)

def get_trading_config() -> TradingConfig:
    return TradingConfig()
