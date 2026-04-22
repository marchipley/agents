import sys
import types
import unittest
from unittest.mock import Mock, patch

import requests

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from custom.btc_agent.llm_decision import decide_trade


class DummyFeatures:
    price_usd = 75000.0
    window_open_price = 74950.0
    delta_pct_from_window_open = 0.000667
    rsi_14 = 55.0
    momentum_5m = 10.0
    volatility_5m = 22.0


class DummyMarket:
    title = "BTC Up or Down"
    slug = "btc-updown-test"


class TestBtcLlmDecision(unittest.TestCase):
    def test_gemini_503_returns_no_trade(self):
        error_response = requests.Response()
        error_response.status_code = 503
        error_response._content = b"service unavailable"

        success_response = requests.Response()
        success_response.status_code = 200
        success_response._content = (
            b'{"candidates":[{"content":{"parts":[{"text":"{\\"decision\\":\\"UP\\",\\"confidence\\":0.8,\\"max_price_to_pay\\":0.6,\\"reason\\":\\"test\\"}"}]}}]}'
        )

        with patch(
            "custom.btc_agent.llm_decision.get_llm_config",
            return_value=Mock(engine="gemini", api_key="test-key", model="gemini-2.5-flash"),
        ), patch(
            "custom.btc_agent.llm_decision.requests.post",
            side_effect=[error_response, success_response],
        ), patch(
            "custom.btc_agent.llm_decision.time.sleep",
        ):
            decision = decide_trade(DummyFeatures(), DummyMarket())

        self.assertEqual(decision.side, "UP")
        self.assertAlmostEqual(decision.confidence, 0.8)

    def test_gemini_total_failure_returns_no_trade(self):
        error_response = requests.Response()
        error_response.status_code = 503
        error_response._content = b"service unavailable"

        with patch(
            "custom.btc_agent.llm_decision.get_llm_config",
            return_value=Mock(engine="gemini", api_key="test-key", model="gemini-2.5-flash"),
        ), patch(
            "custom.btc_agent.llm_decision.requests.post",
            return_value=error_response,
        ), patch(
            "custom.btc_agent.llm_decision.time.sleep",
        ):
            decision = decide_trade(DummyFeatures(), DummyMarket())

        self.assertEqual(decision.side, "NO_TRADE")
        self.assertEqual(decision.confidence, 0.0)
        self.assertIn("LLM request failed", decision.reason)

    def test_gemini_wrapped_text_extracts_json_object(self):
        wrapped_response = requests.Response()
        wrapped_response.status_code = 200
        wrapped_response._content = (
            b'{"candidates":[{"content":{"parts":[{"text":"Here is the decision:\\n{\\"decision\\":\\"DOWN\\",\\"confidence\\":0.71,\\"max_price_to_pay\\":0.42,\\"reason\\":\\"wrapped\\"}\\n"}]}}]}'
        )

        with patch(
            "custom.btc_agent.llm_decision.get_llm_config",
            return_value=Mock(engine="gemini", api_key="test-key", model="gemini-2.5-flash"),
        ), patch(
            "custom.btc_agent.llm_decision.requests.post",
            return_value=wrapped_response,
        ):
            decision = decide_trade(DummyFeatures(), DummyMarket())

        self.assertEqual(decision.side, "DOWN")
        self.assertAlmostEqual(decision.confidence, 0.71)
        self.assertAlmostEqual(decision.max_price_to_pay, 0.42)

    def test_gemini_parse_retry_recovers_from_markdown_preamble(self):
        bad_response = requests.Response()
        bad_response.status_code = 200
        bad_response._content = (
            b'{"candidates":[{"content":{"parts":[{"text":"Here is the JSON requested:\\n```"}]}}]}'
        )

        good_response = requests.Response()
        good_response.status_code = 200
        good_response._content = (
            b'{"candidates":[{"content":{"parts":[{"text":"{\\"decision\\":\\"UP\\",\\"confidence\\":0.74,\\"max_price_to_pay\\":0.31,\\"reason\\":\\"retry ok\\"}"}]}}]}'
        )

        with patch(
            "custom.btc_agent.llm_decision.get_llm_config",
            return_value=Mock(engine="gemini", api_key="test-key", model="gemini-2.5-flash"),
        ), patch(
            "custom.btc_agent.llm_decision.requests.post",
            side_effect=[bad_response, good_response],
        ):
            decision = decide_trade(DummyFeatures(), DummyMarket())

        self.assertEqual(decision.side, "UP")
        self.assertAlmostEqual(decision.confidence, 0.74)

    def test_gemini_truncated_json_retries_with_short_prompt(self):
        truncated_response = requests.Response()
        truncated_response.status_code = 200
        truncated_response._content = (
            b'{"candidates":[{"content":{"parts":[{"text":"{\\"decision\\":\\"NO_TRADE\\",\\"confidence\\":0.0,\\"max_price_to_pay\\":0"}]}}]}'
        )

        recovered_response = requests.Response()
        recovered_response.status_code = 200
        recovered_response._content = (
            b'{"candidates":[{"content":{"parts":[{"text":"{\\"decision\\":\\"NO_TRADE\\",\\"confidence\\":0.0,\\"max_price_to_pay\\":0.0,\\"reason\\":\\"recovered\\"}"}]}}]}'
        )

        with patch(
            "custom.btc_agent.llm_decision.get_llm_config",
            return_value=Mock(engine="gemini", api_key="test-key", model="gemini-2.5-flash"),
        ), patch(
            "custom.btc_agent.llm_decision.requests.post",
            side_effect=[truncated_response, recovered_response],
        ):
            decision = decide_trade(DummyFeatures(), DummyMarket())

        self.assertEqual(decision.side, "NO_TRADE")
        self.assertEqual(decision.confidence, 0.0)
        self.assertEqual(decision.reason, "recovered")

    def test_gemini_read_timeout_retries_and_recovers(self):
        success_response = requests.Response()
        success_response.status_code = 200
        success_response._content = (
            b'{"candidates":[{"content":{"parts":[{"text":"{\\"decision\\":\\"UP\\",\\"confidence\\":0.77,\\"max_price_to_pay\\":0.58,\\"reason\\":\\"timeout recovered\\"}"}]}}]}'
        )

        with patch(
            "custom.btc_agent.llm_decision.get_llm_config",
            return_value=Mock(
                engine="gemini",
                api_key="test-key",
                model="gemini-2.5-flash",
                gemini_connect_timeout_seconds=11.0,
                gemini_read_timeout_seconds=61.0,
                gemini_max_attempts=4,
                gemini_retry_backoff_seconds=2.0,
            ),
        ), patch(
            "custom.btc_agent.llm_decision.requests.post",
            side_effect=[requests.ReadTimeout("read timed out"), success_response],
        ) as mock_post, patch(
            "custom.btc_agent.llm_decision.time.sleep",
        ):
            decision = decide_trade(DummyFeatures(), DummyMarket())

        self.assertEqual(decision.side, "UP")
        self.assertAlmostEqual(decision.confidence, 0.77)
        self.assertEqual(mock_post.call_args_list[0].kwargs["timeout"], (11.0, 61.0))


if __name__ == "__main__":
    unittest.main()
