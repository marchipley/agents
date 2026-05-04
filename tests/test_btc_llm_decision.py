import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import types
import sys

import requests

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object, create_connection=object))

from custom.btc_agent.llm_decision import (
    _build_user_prompt,
    _get_openai_realtime_client,
    _stream_openai_chat_completion,
    decide_trade,
)


class DummyFeatures:
    price_usd = 75000.0
    window_open_price = 74950.0
    trailing_5m_open_price = 74940.0
    delta_pct_from_window_open = 0.000667
    delta_pct_from_trailing_5m_open = 0.000801
    delta_from_previous_tick = 5.0
    rsi_9 = 61.0
    rsi_14 = 55.0
    rsi_speed_divergence = 6.0
    momentum_1m = 7.0
    momentum_5m = 10.0
    velocity_15s = 4.0
    velocity_30s = 6.0
    momentum_acceleration = -2.0
    ema_9 = 74980.0
    ema_21 = 74960.0
    ema_alignment = True
    ema_cross_direction = "bullish"
    adx_14 = 31.0
    atr_14 = 12.0
    volatility_5m = 22.0
    consecutive_flat_ticks = 0
    consecutive_directional_ticks = 3
    as_of = datetime.fromtimestamp(1777513792, tz=timezone.utc)


class DummyMarket:
    title = "BTC Up or Down"
    slug = "btc-updown-test"
    settlement_threshold = 74982.25
    end_ts = 1777513800


class TestBtcLlmDecision(unittest.TestCase):
    def test_get_openai_realtime_client_reuses_existing_client(self):
        fake_client = Mock()
        fake_client.api_key = "test-key"
        fake_client.model = "gpt-realtime-mini"

        with patch(
            "custom.btc_agent.llm_decision._OPENAI_REALTIME_CLIENT",
            fake_client,
        ), patch(
            "custom.btc_agent.llm_decision.OpenAIRealtimeClient",
        ) as mock_client_cls:
            client = _get_openai_realtime_client(
                api_key="test-key",
                model="gpt-4.1-mini",
                timeout_seconds=15.0,
            )

        self.assertIs(client, fake_client)
        mock_client_cls.assert_not_called()

    def test_stream_openai_chat_completion_reassembles_sse_content(self):
        fake_response = Mock()
        fake_response.raise_for_status = Mock()
        fake_response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"{\\"decision\\":\\"UP\\","}}]}',
            'data: {"choices":[{"delta":{"content":"\\"confidence\\":0.8,\\"max_price_to_pay\\":0.5,\\"reason\\":\\"ok\\"}"}}]}',
            "data: [DONE]",
        ]
        fake_response.__enter__ = Mock(return_value=fake_response)
        fake_response.__exit__ = Mock(return_value=None)

        fake_session = Mock()
        fake_session.trust_env = True
        fake_session.post.return_value = fake_response
        fake_session.close = Mock()

        with patch(
            "custom.btc_agent.llm_decision.requests.Session",
            return_value=fake_session,
        ):
            content = _stream_openai_chat_completion(
                model="gpt-4.1-mini",
                api_key="test-key",
                system_prompt="system",
                user_prompt="user",
                timeout_seconds=15.0,
            )

        self.assertEqual(
            content,
            '{"decision":"UP","confidence":0.8,"max_price_to_pay":0.5,"reason":"ok"}',
        )
        self.assertFalse(fake_session.trust_env)
        self.assertTrue(fake_session.post.call_args.kwargs["stream"])

    def test_user_prompt_includes_price_to_beat(self):
        up_snapshot = Mock(buy_quote=0.84)
        down_snapshot = Mock(buy_quote=0.17)
        prompt = _build_user_prompt(
            DummyFeatures(),
            DummyMarket(),
            up_snapshot=up_snapshot,
            down_snapshot=down_snapshot,
        )

        self.assertIn("Price to beat USD: 74982.25", prompt)
        self.assertIn("Market reference:", prompt)
        self.assertIn("UP wins only if BTC finishes above 74982.25", prompt)
        self.assertIn("DOWN wins only if BTC finishes below 74982.25", prompt)
        self.assertIn("Time remaining seconds: 8", prompt)
        self.assertIn("UP Polymarket ask/buy quote: 0.84", prompt)
        self.assertIn("DOWN Polymarket ask/buy quote: 0.17", prompt)
        self.assertIn("RSI(9): 61.0", prompt)
        self.assertIn("ADX(14): 31.0", prompt)
        self.assertIn("Trend intensity (ADX): 31.0", prompt)
        self.assertIn("EMA alignment (Price > EMA9 > EMA21): True", prompt)
        self.assertIn("Momentum acceleration: -2.0", prompt)
        self.assertIn("Oracle gap ratio:", prompt)
        self.assertIn("time_remaining_seconds is authoritative", prompt)
        self.assertIn("Discovery Phase", prompt)
        self.assertIn("Do not confuse it with Oracle gap ratio", prompt)
        self.assertIn("window_delta_pct is the source of truth for overall trend direction", prompt)
        self.assertIn("velocity_30s is micro-momentum for entry timing only", prompt)
        self.assertIn("If the chosen side buy quote is below 0.15", prompt)
        self.assertIn("If window_delta_pct is positive and you choose DOWN", prompt)
        self.assertIn("Focus on regime detection and direction, not limit pricing.", prompt)
        self.assertIn("execution layer will apply regime-aware EV, deadline, liquidity, and FOK rules", prompt)

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
            return_value=Mock(
                engine="gemini",
                api_key="test-key",
                model="gemini-2.5-flash",
                api_connection_timeout_seconds=10.0,
                api_connection_retry_timer_seconds=2.0,
                api_connection_retry_attempts=3,
            ),
        ), patch(
            "custom.btc_agent.llm_decision._direct_http_post",
            side_effect=[error_response, success_response],
        ), patch(
            "custom.btc_agent.llm_decision.check_internet_connectivity",
            return_value=(True, "Connectivity OK via https://www.google.com/generate_204 (HTTP 204)"),
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
            return_value=Mock(
                engine="gemini",
                api_key="test-key",
                model="gemini-2.5-flash",
                api_connection_timeout_seconds=10.0,
                api_connection_retry_timer_seconds=2.0,
                api_connection_retry_attempts=3,
            ),
        ), patch(
            "custom.btc_agent.llm_decision._direct_http_post",
            return_value=error_response,
        ), patch(
            "custom.btc_agent.llm_decision.check_internet_connectivity",
            return_value=(True, "Connectivity OK via https://www.google.com/generate_204 (HTTP 204)"),
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
            return_value=Mock(
                engine="gemini",
                api_key="test-key",
                model="gemini-2.5-flash",
                api_connection_timeout_seconds=10.0,
                api_connection_retry_timer_seconds=2.0,
                api_connection_retry_attempts=3,
            ),
        ), patch(
            "custom.btc_agent.llm_decision._direct_http_post",
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
            return_value=Mock(
                engine="gemini",
                api_key="test-key",
                model="gemini-2.5-flash",
                api_connection_timeout_seconds=10.0,
                api_connection_retry_timer_seconds=2.0,
                api_connection_retry_attempts=3,
            ),
        ), patch(
            "custom.btc_agent.llm_decision._direct_http_post",
            side_effect=[bad_response, good_response],
        ), patch(
            "builtins.print",
        ) as mock_print:
            decision = decide_trade(DummyFeatures(), DummyMarket())

        printed_lines = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
        self.assertTrue(any("[invalid-json] failed" in line for line in printed_lines))
        self.assertTrue(any("LLM attempt 2/3 (gemini/gemini-2.5-flash) response" in line for line in printed_lines))
        self.assertEqual(decision.side, "UP")
        self.assertAlmostEqual(decision.confidence, 0.74)

    def test_gemini_logs_connection_proxy_and_timeout(self):
        success_response = requests.Response()
        success_response.status_code = 200
        success_response._content = (
            b'{"candidates":[{"content":{"parts":[{"text":"{\\"decision\\":\\"UP\\",\\"confidence\\":0.8,\\"max_price_to_pay\\":0.6,\\"reason\\":\\"test\\"}"}]}}]}'
        )

        with patch(
            "custom.btc_agent.llm_decision.get_llm_config",
            return_value=Mock(
                engine="gemini",
                api_key="test-key",
                model="gemini-2.5-flash",
                api_connection_timeout_seconds=15.0,
                api_connection_retry_timer_seconds=2.0,
                api_connection_retry_attempts=1,
            ),
        ), patch(
            "custom.btc_agent.llm_decision._direct_http_post",
            return_value=success_response,
        ), patch(
            "builtins.print",
        ) as mock_print:
            decide_trade(DummyFeatures(), DummyMarket())

        printed_lines = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
        self.assertTrue(any(line == "LLM connection:" for line in printed_lines))
        self.assertTrue(any("engine            = gemini" in line for line in printed_lines))
        self.assertTrue(any("model             = gemini-2.5-flash" in line for line in printed_lines))
        self.assertTrue(any("timeout_seconds   = 15.0" in line for line in printed_lines))
        self.assertTrue(any("proxy             = None" in line for line in printed_lines))

    def test_openai_disables_trust_env_when_use_proxy_false(self):
        with patch(
            "custom.btc_agent.llm_decision.get_llm_config",
            return_value=Mock(
                engine="openai",
                api_key="test-key",
                model="gpt-4.1-mini",
                api_connection_timeout_seconds=15.0,
                api_connection_retry_timer_seconds=2.0,
                api_connection_retry_attempts=1,
            ),
        ), patch(
            "custom.btc_agent.llm_decision._get_openai_realtime_client",
            return_value=Mock(
                request=Mock(
                    return_value='{"decision":"UP","confidence":0.8,"max_price_to_pay":0.5,"reason":"ok"}'
                )
            ),
        ):
            decision = decide_trade(DummyFeatures(), DummyMarket())

        self.assertEqual(decision.side, "UP")

    def test_openai_retries_same_minimal_prompt_after_connection_error(self):
        with patch(
            "custom.btc_agent.llm_decision.get_llm_config",
            return_value=Mock(
                engine="openai",
                api_key="test-key",
                model="gpt-4.1-mini",
                api_connection_timeout_seconds=15.0,
                api_connection_retry_timer_seconds=2.0,
                api_connection_retry_attempts=2,
            ),
        ), patch(
            "custom.btc_agent.llm_decision._request_openai_once",
            side_effect=[
                requests.ConnectionError("Connection reset by peer"),
                '{"decision":"UP","confidence":0.81,"max_price_to_pay":0.55,"reason":"compact ok"}',
            ],
        ) as mock_request_openai_once, patch(
            "custom.btc_agent.llm_decision.check_internet_connectivity",
            return_value=(True, "Connectivity OK via https://www.google.com/generate_204 (HTTP 204)"),
        ), patch(
            "builtins.print",
        ) as mock_print:
            decision = decide_trade(DummyFeatures(), DummyMarket())

        printed_lines = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
        self.assertEqual(decision.side, "UP")
        self.assertEqual(mock_request_openai_once.call_count, 2)
        self.assertEqual(
            mock_request_openai_once.call_args_list[0].kwargs["user_prompt"],
            mock_request_openai_once.call_args_list[1].kwargs["user_prompt"],
        )
        self.assertFalse(any("[fallback]" in line for line in printed_lines))

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
            return_value=Mock(
                engine="gemini",
                api_key="test-key",
                model="gemini-2.5-flash",
                api_connection_timeout_seconds=10.0,
                api_connection_retry_timer_seconds=2.0,
                api_connection_retry_attempts=3,
            ),
        ), patch(
            "custom.btc_agent.llm_decision._direct_http_post",
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
                api_connection_timeout_seconds=11.0,
                api_connection_retry_timer_seconds=2.0,
                api_connection_retry_attempts=4,
            ),
        ), patch(
            "custom.btc_agent.llm_decision._direct_http_post",
            side_effect=[requests.ReadTimeout("read timed out"), success_response],
        ) as mock_requests_post, patch(
            "custom.btc_agent.llm_decision.check_internet_connectivity",
            return_value=(True, "Connectivity OK via https://www.google.com/generate_204 (HTTP 204)"),
        ), patch(
            "custom.btc_agent.llm_decision.time.sleep",
        ):
            decision = decide_trade(DummyFeatures(), DummyMarket())

        self.assertEqual(decision.side, "UP")
        self.assertAlmostEqual(decision.confidence, 0.77)
        self.assertEqual(mock_requests_post.call_args_list[0].kwargs["timeout"], 11.0)

    def test_gemini_incomplete_json_retries_full_attempts_then_fails(self):
        truncated_response = requests.Response()
        truncated_response.status_code = 200
        truncated_response._content = (
            b'{"candidates":[{"content":{"parts":[{"text":"{\\"decision\\":\\"UP\\",\\"confidence\\":0.6"}]}}]}'
        )

        with patch(
            "custom.btc_agent.llm_decision.get_llm_config",
            return_value=Mock(
                engine="gemini",
                api_key="test-key",
                model="gemini-3.1-pro-preview",
                api_connection_timeout_seconds=10.0,
                api_connection_retry_timer_seconds=1.0,
                api_connection_retry_attempts=2,
            ),
        ), patch(
            "custom.btc_agent.llm_decision._direct_http_post",
            side_effect=[truncated_response, truncated_response],
        ), patch(
            "custom.btc_agent.llm_decision.time.sleep",
        ), patch(
            "builtins.print",
        ) as mock_print:
            decision = decide_trade(DummyFeatures(), DummyMarket())

        printed_lines = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
        self.assertTrue(any("LLM attempt 1/2 (gemini/gemini-3.1-pro-preview) response" in line for line in printed_lines))
        self.assertTrue(any("LLM attempt 1/2 (gemini/gemini-3.1-pro-preview) [invalid-json] failed" in line for line in printed_lines))
        self.assertTrue(any("LLM attempt 2/2 (gemini/gemini-3.1-pro-preview) response" in line for line in printed_lines))
        self.assertTrue(any("LLM attempt 2/2 (gemini/gemini-3.1-pro-preview) [invalid-json] failed" in line for line in printed_lines))
        self.assertEqual(decision.side, "NO_TRADE")
        self.assertIn("LLM request failed", decision.reason)

    def test_gemini_logs_each_attempt_and_returns_no_trade_after_final_failure(self):
        with patch(
            "custom.btc_agent.llm_decision.get_llm_config",
            return_value=Mock(
                engine="gemini",
                api_key="test-key",
                model="gemini-2.5-flash",
                api_connection_timeout_seconds=10.0,
                api_connection_retry_timer_seconds=1.0,
                api_connection_retry_attempts=2,
            ),
        ), patch(
            "custom.btc_agent.llm_decision._direct_http_post",
            side_effect=[
                requests.ReadTimeout("first timeout"),
                requests.ReadTimeout("second timeout"),
            ],
        ), patch(
            "custom.btc_agent.llm_decision.check_internet_connectivity",
            return_value=(True, "Connectivity OK via https://www.google.com/generate_204 (HTTP 204)"),
        ), patch(
            "custom.btc_agent.llm_decision.time.sleep",
        ), patch(
            "builtins.print",
        ) as mock_print:
            decision = decide_trade(DummyFeatures(), DummyMarket())

        printed_lines = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
        self.assertTrue(any("LLM attempt 1/2 (gemini/gemini-2.5-flash) failed" in line for line in printed_lines))
        self.assertTrue(any("LLM attempt 2/2 (gemini/gemini-2.5-flash) failed" in line for line in printed_lines))
        self.assertEqual(decision.side, "NO_TRADE")

    def test_gemini_connection_failure_runs_connectivity_check_before_retry(self):
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
                api_connection_timeout_seconds=11.0,
                api_connection_retry_timer_seconds=2.0,
                api_connection_retry_attempts=2,
            ),
        ), patch(
            "custom.btc_agent.llm_decision._direct_http_post",
            side_effect=[requests.ReadTimeout("read timed out"), success_response],
        ), patch(
            "custom.btc_agent.llm_decision.check_internet_connectivity",
            return_value=(True, "Connectivity OK via https://www.google.com/generate_204 (HTTP 204)"),
        ) as mock_connectivity, patch(
            "custom.btc_agent.llm_decision.time.sleep",
        ), patch(
            "builtins.print",
        ) as mock_print:
            decision = decide_trade(DummyFeatures(), DummyMarket())

        printed_lines = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
        self.assertEqual(decision.side, "UP")
        mock_connectivity.assert_called_once()
        self.assertTrue(any("Internet connectivity check: Connectivity OK" in line for line in printed_lines))

    def test_gemini_connection_failure_stops_when_connectivity_check_fails(self):
        with patch(
            "custom.btc_agent.llm_decision.get_llm_config",
            return_value=Mock(
                engine="gemini",
                api_key="test-key",
                model="gemini-2.5-flash",
                api_connection_timeout_seconds=11.0,
                api_connection_retry_timer_seconds=2.0,
                api_connection_retry_attempts=3,
            ),
        ), patch(
            "custom.btc_agent.llm_decision._direct_http_post",
            side_effect=requests.ReadTimeout("read timed out"),
        ), patch(
            "custom.btc_agent.llm_decision.check_internet_connectivity",
            return_value=(False, "Connectivity check failed via https://www.google.com/generate_204: offline"),
        ) as mock_connectivity, patch(
            "custom.btc_agent.llm_decision.time.sleep",
        ) as mock_sleep:
            decision = decide_trade(DummyFeatures(), DummyMarket())

        self.assertEqual(decision.side, "NO_TRADE")
        self.assertIn("Connectivity check failed", decision.reason)
        mock_connectivity.assert_called_once()
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
