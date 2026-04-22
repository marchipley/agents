import os
import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))

from custom.btc_agent.config import get_llm_config


class TestBtcLlmConfig(unittest.TestCase):
    def test_openai_engine_uses_openai_key_and_model(self):
        with patch.dict(
            os.environ,
            {
                "AI_ENGINE": "OPENAI",
                "OPENAI_API_KEY": "openai-key",
                "OPENAI_MODEL": "gpt-4.1-mini",
            },
            clear=False,
        ):
            cfg = get_llm_config()

        self.assertEqual(cfg.engine, "openai")
        self.assertEqual(cfg.api_key, "openai-key")
        self.assertEqual(cfg.model, "gpt-4.1-mini")

    def test_gemini_engine_uses_gemini_key_and_model(self):
        with patch.dict(
            os.environ,
            {
                "AI_ENGINE": "GEMINI",
                "GEMINI_API_KEY": "gemini-key",
                "GEMINI_MODEL": "gemini-2.5-flash",
            },
            clear=False,
        ):
            cfg = get_llm_config()

        self.assertEqual(cfg.engine, "gemini")
        self.assertEqual(cfg.api_key, "gemini-key")
        self.assertEqual(cfg.model, "gemini-2.5-flash")

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


if __name__ == "__main__":
    unittest.main()
