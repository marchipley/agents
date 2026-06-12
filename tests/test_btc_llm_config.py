import os
import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))

from custom.btc_agent.config import get_llm_config, get_trading_config


class TestBtcLlmConfig(unittest.TestCase):
    def test_openai_engine_uses_openai_key_and_model(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
            },
            clear=False,
        ), patch("custom.btc_agent.config.AI_ENGINE", "OPENAI"), patch(
            "custom.btc_agent.config.OPENAI_MODEL",
            "gpt-4.1-mini",
        ):
            cfg = get_llm_config()

        self.assertEqual(cfg.engine, "openai")
        self.assertEqual(cfg.api_key, "openai-key")
        self.assertEqual(cfg.model, "gpt-4.1-mini")

    def test_gemini_engine_uses_gemini_key_and_model(self):
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "gemini-key",
            },
            clear=False,
        ), patch("custom.btc_agent.config.AI_ENGINE", "GEMINI"), patch(
            "custom.btc_agent.config.GEMINI_MODEL",
            "gemini-3.1-flash-lite-preview",
        ):
            cfg = get_llm_config()

        self.assertEqual(cfg.engine, "gemini")
        self.assertEqual(cfg.api_key, "gemini-key")
        self.assertEqual(cfg.model, "gemini-3.1-flash-lite-preview")
        self.assertEqual(cfg.api_connection_timeout_seconds, 10.0)
        self.assertEqual(cfg.api_connection_retry_timer_seconds, 2.0)
        self.assertEqual(cfg.api_connection_retry_attempts, 3)

    def test_engine_uses_api_connection_overrides(self):
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "gemini-key",
                "API_CONNECTION_TIMEOUT": "7",
                "API_CONNECTION_RETRY_TIMER": "3.5",
                "API_CONNECTION_RETRY_ATTEMPTS": "5",
            },
            clear=False,
        ), patch("custom.btc_agent.config.AI_ENGINE", "GEMINI"), patch(
            "custom.btc_agent.config.GEMINI_MODEL",
            "gemini-3.1-flash-lite-preview",
        ):
            cfg = get_llm_config()

        self.assertEqual(cfg.api_connection_timeout_seconds, 7.0)
        self.assertEqual(cfg.api_connection_retry_timer_seconds, 3.5)
        self.assertEqual(cfg.api_connection_retry_attempts, 5)

    def test_unknown_engine_raises(self):
        with patch.dict(
            os.environ,
            {},
            clear=False,
        ), patch("custom.btc_agent.config.AI_ENGINE", "ANTHROPIC"):
            with self.assertRaises(RuntimeError):
                get_llm_config()

    def test_trading_config_reads_llm_connection_debug_flag_from_config_constant(self):
        with patch("custom.btc_agent.config.LLM_CONNECTION_DEBUG", True):
            cfg = get_trading_config()

        self.assertTrue(cfg.llm_connection_debug)

    def test_trading_config_reads_llm_show_detail_flag_from_config_constant(self):
        with patch("custom.btc_agent.config.LLM_SHOW_DETAIL", True):
            cfg = get_trading_config()

        self.assertTrue(cfg.llm_show_detail)

    def test_trading_config_reads_max_allowable_price_from_config_constant(self):
        with patch("custom.btc_agent.config.MAX_ALLOWABLE_PRICE", 0.95):
            cfg = get_trading_config()

        self.assertEqual(cfg.max_allowable_price, 0.95)


if __name__ == "__main__":
    unittest.main()
