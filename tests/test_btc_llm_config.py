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
                "AI_ENGINE": "OPENAI",
                "OPENAI_API_KEY": "openai-key",
                "OPENAI_MODEL": "gpt-4.1-mini",
                "LLM_PROXY_URL": "",
                "LLM_WINDOWS_HOST_IP": "",
                "LLM_PROXY_PORT": "",
            },
            clear=False,
        ):
            cfg = get_llm_config()

        self.assertEqual(cfg.engine, "openai")
        self.assertEqual(cfg.api_key, "openai-key")
        self.assertEqual(cfg.model, "gpt-4.1-mini")
        self.assertEqual(cfg.proxy_url, "http://10.255.255.254:8888")

    def test_gemini_engine_uses_gemini_key_and_model(self):
        with patch.dict(
            os.environ,
            {
                "AI_ENGINE": "GEMINI",
                "GEMINI_API_KEY": "gemini-key",
                "GEMINI_MODEL": "gemini-2.5-flash",
                "LLM_PROXY_URL": "",
                "LLM_WINDOWS_HOST_IP": "",
                "LLM_PROXY_PORT": "",
            },
            clear=False,
        ):
            cfg = get_llm_config()

        self.assertEqual(cfg.engine, "gemini")
        self.assertEqual(cfg.api_key, "gemini-key")
        self.assertEqual(cfg.model, "gemini-2.5-flash")
        self.assertEqual(cfg.proxy_url, "http://10.255.255.254:8888")
        self.assertEqual(cfg.api_connection_timeout_seconds, 10.0)
        self.assertEqual(cfg.api_connection_retry_timer_seconds, 2.0)
        self.assertEqual(cfg.api_connection_retry_attempts, 3)

    def test_llm_proxy_url_override_is_used(self):
        with patch.dict(
            os.environ,
            {
                "AI_ENGINE": "GEMINI",
                "GEMINI_API_KEY": "gemini-key",
                "LLM_WINDOWS_HOST_IP": "",
                "LLM_PROXY_PORT": "",
                "LLM_PROXY_URL": "http://10.1.2.3:9999",
            },
            clear=False,
        ):
            cfg = get_llm_config()

        self.assertEqual(cfg.proxy_url, "http://10.1.2.3:9999")

    def test_engine_uses_api_connection_overrides(self):
        with patch.dict(
            os.environ,
            {
                "AI_ENGINE": "GEMINI",
                "GEMINI_API_KEY": "gemini-key",
                "GEMINI_MODEL": "gemini-2.5-flash",
                "API_CONNECTION_TIMEOUT": "7",
                "LLM_API_READ_TIMEOUT": "19",
                "API_CONNECTION_RETRY_TIMER": "3.5",
                "API_CONNECTION_RETRY_ATTEMPTS": "5",
            },
            clear=False,
        ):
            cfg = get_llm_config()

        self.assertEqual(cfg.api_connection_timeout_seconds, 7.0)
        self.assertEqual(cfg.api_read_timeout_seconds, 19.0)
        self.assertEqual(cfg.api_connection_retry_timer_seconds, 3.5)
        self.assertEqual(cfg.api_connection_retry_attempts, 5)

    def test_unknown_engine_raises(self):
        with patch.dict(
            os.environ,
            {
                "AI_ENGINE": "ANTHROPIC",
            },
            clear=False,
        ):
            with self.assertRaises(RuntimeError):
                get_llm_config()

    def test_trading_config_reads_llm_connection_debug_flag(self):
        with patch.dict(
            os.environ,
            {
                "LLM_CONNECTION_DEBUG": "true",
            },
            clear=False,
        ):
            cfg = get_trading_config()

        self.assertTrue(cfg.llm_connection_debug)

    def test_trading_config_reads_llm_response_debug_flag_from_repo_constant(self):
        with patch(
            "custom.btc_agent.config.LLM_RESPONSE_DEBUG",
            True,
        ):
            cfg = get_trading_config()

        self.assertTrue(cfg.llm_response_debug)

    def test_trading_config_uses_repo_debug_flag_and_forces_paper_trading(self):
        with patch.dict(
            os.environ,
            {
                "USE_PAPER_TRADES": "false",
            },
            clear=False,
        ), patch(
            "custom.btc_agent.config.BTC_AGENT_DEBUG",
            True,
        ):
            cfg = get_trading_config()

        self.assertTrue(cfg.debug)
        self.assertTrue(cfg.paper_trading)


if __name__ == "__main__":
    unittest.main()
