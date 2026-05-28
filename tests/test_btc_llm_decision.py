import unittest
import json
import socket
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import types
import sys

import requests

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object, create_connection=object))

from custom.btc_agent.llm_decision import (
    _build_debug_prompt_text,
    _build_minimal_user_prompt,
    _build_openai_realtime_user_prompt,
    _build_system_prompt,
    _build_user_prompt,
    _extract_json_payload,
    _get_openai_realtime_client,
    _get_openai_rest_session,
    OpenAIRealtimeClient,
    _request_openai_once,
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
    last_10_ticks_direction = "UUUDUUUUUU"
    as_of = datetime.fromtimestamp(1777513792, tz=timezone.utc)


class DummyMarket:
    title = "BTC Up or Down"
    slug = "btc-updown-test"
    settlement_threshold = 74982.25
    end_ts = 1777513800
    start_ts = 1777513500
    up_market_probability = 0.495
    down_market_probability = 0.505


class TestBtcLlmDecision(unittest.TestCase):
    def test_build_debug_prompt_text_enabled_by_llm_show_detail_without_full_debug(self):
        with patch(
            "custom.btc_agent.llm_decision.get_trading_config",
            return_value=types.SimpleNamespace(debug=False, llm_show_detail=True),
        ):
            prompt_text = _build_debug_prompt_text("system", "user")

        self.assertEqual(prompt_text, "SYSTEM PROMPT:\nsystem\n\nUSER PROMPT:\nuser")

    def test_extract_json_payload_accepts_key_value_response_format(self):
        payload = _extract_json_payload(
            "decision: NO_TRADE, confidence: 0.45, max_price_to_pay: 1.0, reason: Time remaining is sufficient, RSI not extreme, and side quote low, so prefer no trade."
        )

        self.assertEqual(payload["decision"], "NO_TRADE")
        self.assertEqual(payload["confidence"], 0.45)
        self.assertEqual(payload["max_price_to_pay"], 1.0)
        self.assertIn("prefer no trade", payload["reason"].lower())

    def test_extract_json_payload_accepts_multiline_key_value_response_without_commas(self):
        payload = _extract_json_payload(
            "decision: DOWN\nconfidence: 0.72\nmax price to pay: 1.0\nreason: Price is below strike\nand momentum agrees"
        )

        self.assertEqual(payload["decision"], "DOWN")
        self.assertEqual(payload["confidence"], 0.72)
        self.assertEqual(payload["max_price_to_pay"], 1.0)
        self.assertIn("momentum agrees", payload["reason"])

    def test_extract_json_payload_accepts_quoted_key_value_response_with_semicolons(self):
        payload = _extract_json_payload(
            "\"decision\": \"UP\"; \"confidence\": \"0.81\"; \"max_price_to_pay\": \"1.0\"; \"reason\": \"Breakout continuation\""
        )

        self.assertEqual(payload["decision"], "UP")
        self.assertEqual(payload["confidence"], 0.81)
        self.assertEqual(payload["max_price_to_pay"], 1.0)
        self.assertIn("Breakout", payload["reason"])

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

    def test_openai_realtime_client_uses_ga_session_schema(self):
        sent_payloads = []

        class FakeRealtimeSocket:
            def __init__(self):
                self.messages = [
                    json.dumps(
                        {
                            "type": "response.output_text.delta",
                            "delta": '{"decision":"UP","confidence":0.8,',
                        }
                    ),
                    json.dumps(
                        {
                            "type": "response.output_text.delta",
                            "delta": '"max_price_to_pay":1.0,"reason":"ok"}',
                        }
                    ),
                    json.dumps({"type": "response.done"}),
                ]

            def settimeout(self, _timeout):
                pass

            def send(self, payload):
                sent_payloads.append(json.loads(payload))

            def recv(self):
                return self.messages.pop(0)

            def close(self):
                pass

        with patch(
            "custom.btc_agent.llm_decision.websocket.create_connection",
            return_value=FakeRealtimeSocket(),
        ) as create_connection:
            client = OpenAIRealtimeClient("test-key", "gpt-realtime-mini", 15.0)
            response = client.request("system", "user")

        self.assertIn('"decision":"UP"', response)
        create_connection.assert_called_once()
        connect_kwargs = create_connection.call_args.kwargs
        self.assertNotIn("OpenAI-Beta: realtime=v1", connect_kwargs["header"])
        self.assertIn((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1), connect_kwargs["sockopt"])
        self.assertIn((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1), connect_kwargs["sockopt"])
        session_update = sent_payloads[0]
        response_create = sent_payloads[2]
        self.assertEqual(session_update["type"], "session.update")
        self.assertEqual(session_update["session"]["type"], "realtime")
        self.assertEqual(session_update["session"]["output_modalities"], ["text"])
        self.assertEqual(session_update["session"]["instructions"], "system")
        self.assertNotIn("modalities", session_update["session"])
        self.assertNotIn("temperature", session_update["session"])
        self.assertEqual(response_create["type"], "response.create")
        self.assertNotIn("modalities", response_create["response"])
        self.assertEqual(response_create["response"]["max_output_tokens"], 512)

    def test_openai_realtime_client_reconnects_after_idle_window(self):
        class FakeRealtimeSocket:
            def __init__(self, text):
                self.messages = [
                    json.dumps({"type": "response.output_text.delta", "delta": text}),
                    json.dumps({"type": "response.done"}),
                ]
                self.timeout_values = []
                self.closed = False

            def settimeout(self, timeout):
                self.timeout_values.append(timeout)

            def send(self, _payload):
                pass

            def recv(self):
                return self.messages.pop(0)

            def close(self):
                self.closed = True

        sockets = [FakeRealtimeSocket("first"), FakeRealtimeSocket("second")]

        with patch(
            "custom.btc_agent.llm_decision.websocket.create_connection",
            side_effect=sockets,
        ) as create_connection:
            client = OpenAIRealtimeClient("test-key", "gpt-realtime-mini", 15.0)
            self.assertEqual(client.request("system", "user"), "first")
            client._last_request_time = time.monotonic() - 13.0
            self.assertEqual(client.request("system", "user"), "second")

        self.assertEqual(create_connection.call_count, 2)
        self.assertTrue(sockets[0].closed)
        self.assertIn(1.0, sockets[0].timeout_values)

    def test_openai_realtime_client_closes_on_empty_frame(self):
        class FakeRealtimeSocket:
            def __init__(self):
                self.closed = False

            def settimeout(self, _timeout):
                pass

            def send(self, _payload):
                pass

            def recv(self):
                return ""

            def close(self):
                self.closed = True

        socket = FakeRealtimeSocket()

        with patch(
            "custom.btc_agent.llm_decision.websocket.create_connection",
            return_value=socket,
        ):
            client = OpenAIRealtimeClient("test-key", "gpt-realtime-mini", 15.0)
            with self.assertRaisesRegex(RuntimeError, "WebSocket closed unexpectedly"):
                client.request("system", "user")

        self.assertTrue(socket.closed)
        self.assertIsNone(client.ws)

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
        self.assertEqual(fake_session.post.call_args.kwargs["json"]["max_tokens"], 100)

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
        self.assertIn("DISTANCE_FROM_STRIKE_PCT:", prompt)
        self.assertIn("Current BTC price USD (raw feed): 75000.00", prompt)
        self.assertIn("Effective BTC price USD (raw strike baseline):", prompt)
        self.assertIn("Implied oracle price USD (market context only):", prompt)
        self.assertIn("UP Polymarket ask/buy quote: 0.84", prompt)
        self.assertIn("DOWN Polymarket ask/buy quote: 0.17", prompt)
        self.assertIn("RSI(9): 61.0", prompt)
        self.assertIn("ADX(14): 31.0", prompt)
        self.assertIn("Trend intensity (ADX): 31.0", prompt)
        self.assertIn("EMA alignment (Price > EMA9 > EMA21): True", prompt)
        self.assertIn("Momentum acceleration: -2.0", prompt)
        self.assertIn("Momentum alignment:", prompt)
        self.assertIn("Oracle gap ratio:", prompt)
        self.assertIn("trend_regime:", prompt)
        self.assertIn("rsi_regime:", prompt)
        self.assertIn("volatility_regime:", prompt)
        self.assertNotIn("Decision policy:", prompt)
        self.assertNotIn("prefer NO_TRADE", prompt)
        self.assertNotIn("do not choose", prompt)

    def test_user_prompt_uses_canonical_window_time_when_end_ts_is_stale(self):
        class StaleEndMarket(DummyMarket):
            end_ts = 1777513505
            start_ts = 1777513500

        up_snapshot = Mock(buy_quote=0.84)
        down_snapshot = Mock(buy_quote=0.17)
        prompt = _build_user_prompt(
            DummyFeatures(),
            StaleEndMarket(),
            up_snapshot=up_snapshot,
            down_snapshot=down_snapshot,
        )

        self.assertIn("Time remaining seconds: 8", prompt)

    def test_user_prompt_prefers_slug_timestamp_when_start_and_end_are_stale(self):
        class StaleTimingMarket(DummyMarket):
            slug = "btc-updown-5m-1777513500"
            start_ts = 1777513200
            end_ts = 1777513210

        up_snapshot = Mock(buy_quote=0.84)
        down_snapshot = Mock(buy_quote=0.17)
        prompt = _build_user_prompt(
            DummyFeatures(),
            StaleTimingMarket(),
            up_snapshot=up_snapshot,
            down_snapshot=down_snapshot,
        )

        self.assertIn("Time remaining seconds: 8", prompt)

    def test_minimal_prompt_uses_gamma_probabilities_and_strike_delta_not_window_delta(self):
        up_snapshot = Mock(buy_quote=0.84)
        down_snapshot = Mock(buy_quote=0.17)
        prompt = _build_minimal_user_prompt(
            DummyFeatures(),
            DummyMarket(),
            up_snapshot=up_snapshot,
            down_snapshot=down_snapshot,
        )

        self.assertIn("DISTANCE_FROM_STRIKE_USD=", prompt)
        self.assertIn("MARKET_WIN_CHANCE_UP=0.495", prompt)
        self.assertIn("MARKET_WIN_CHANCE_DOWN=0.505", prompt)
        self.assertIn("momentum_direction=UP", prompt)
        self.assertIn("momentum_alignment=", prompt)
        self.assertIn("rsi_speed_divergence=", prompt)
        self.assertIn("trend_regime=", prompt)
        self.assertIn("rsi_regime=", prompt)
        self.assertIn("volatility_regime=", prompt)
        self.assertNotIn("prefer NO_TRADE", prompt)
        self.assertNotIn("Do not confuse DISTANCE_FROM_STRIKE values", prompt)
        self.assertNotIn("\ndelta_pct=", prompt)

    def test_openai_realtime_user_prompt_includes_regime_strings_without_policy_suffix(self):
        up_snapshot = Mock(buy_quote=0.84)
        down_snapshot = Mock(buy_quote=0.17)
        prompt = _build_openai_realtime_user_prompt(
            DummyFeatures(),
            DummyMarket(),
            up_snapshot=up_snapshot,
            down_snapshot=down_snapshot,
        )

        self.assertIn("trend_regime=", prompt)
        self.assertIn("rsi_regime=", prompt)
        self.assertIn("volatility_regime=", prompt)
        self.assertIn("momentum_direction=UP;", prompt)
        self.assertIn("momentum_alignment=True;", prompt)
        self.assertNotIn("if_t_gt_", prompt)
        self.assertNotIn("prefer_no_trade", prompt)

    def test_openai_realtime_user_prompt_handles_missing_momentum_as_neutral(self):
        class NoMomentumFeatures(DummyFeatures):
            momentum_1m = None

        up_snapshot = Mock(buy_quote=0.84)
        down_snapshot = Mock(buy_quote=0.17)
        prompt = _build_openai_realtime_user_prompt(
            NoMomentumFeatures(),
            DummyMarket(),
            up_snapshot=up_snapshot,
            down_snapshot=down_snapshot,
        )

        self.assertIn("momentum_direction=NEUTRAL;", prompt)

    def test_system_prompt_contains_moved_policy_rules(self):
        prompt = _build_system_prompt()

        self.assertIn("Parabolic Exception", prompt)
        self.assertIn("Gamma Discrepancy Rule", prompt)
        self.assertIn("CRITICAL VELOCITY RULE", prompt)
        self.assertIn("VOLATILITY RATIO RULE", prompt)
        self.assertIn("EARLY WINDOW CUSHION RULE", prompt)
        self.assertIn("cushions under $50", prompt)
        self.assertIn("LATE WINDOW VOLATILITY PENALTY", prompt)
        self.assertIn("EXHAUSTION REVERSION RULE", prompt)
        self.assertIn("entry quote is < 0.85", prompt)
        self.assertIn("Alpha Override Rule: If confidence is > 0.75", prompt)
        self.assertIn("ITM AGGRESSION RULE: If the trade is ITM by more than $15.00", prompt)
        self.assertIn("Discovery Rule: If ADX < 12.0", prompt)
        self.assertIn("ADX Cap Rule: Treat ADX > 55 as trend exhaustion", prompt)
        self.assertIn("ITM Velocity Rule: If trend_regime is weak", prompt)
        self.assertIn("Weak Regime Rule: If trend_regime contains 'weak' and RSI speed divergence", prompt)
        self.assertIn("STAGNATION RULE: If ADX < 12.0", prompt)
        self.assertIn("Time is of the essence", prompt)
        self.assertIn("reason' field to 1 or 2 short sentences maximum", prompt)

    def test_openai_realtime_prompt_forbids_conversational_filler(self):
        from custom.btc_agent.llm_decision import _build_openai_realtime_system_prompt

        prompt = _build_openai_realtime_system_prompt()

        self.assertIn("Return ONE raw JSON object only", prompt)
        self.assertIn("'decision', 'confidence', 'max_price_to_pay', and 'reason'", prompt)
        self.assertIn("Do NOT output markdown blocks", prompt)
        self.assertIn("do NOT output conversational filler", prompt)
        self.assertIn("If the chosen side MARKET_WIN_CHANCE is greater than 0.90", prompt)
        self.assertIn("If ITM gap is decreasing", prompt)
        self.assertIn("If rsi_regime is labeled PARABOLIC", prompt)
        self.assertIn("trust the price action over the oscillator", prompt)
        self.assertIn("For OTM trades, you must have an execution edge of at least 0.05", prompt)
        self.assertIn("VELOCITY OVERRIDE: If delta_prev_tick is moving against your chosen side", prompt)
        self.assertIn("EARLY WINDOW CUSHION RULE", prompt)
        self.assertIn("cushions under $50", prompt)
        self.assertIn("LATE WINDOW VOLATILITY PENALTY", prompt)
        self.assertIn("EXHAUSTION REVERSION RULE", prompt)
        self.assertIn("Time is of the essence", prompt)
        self.assertIn("reason' field to 1 or 2 short sentences maximum", prompt)

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

    def test_openai_uses_rest_chat_completions_session(self):
        response = Mock(
            ok=True,
            json=Mock(
                return_value={
                    "choices": [
                        {
                            "message": {
                                "content": '{"decision":"UP","confidence":0.8,"max_price_to_pay":0.5,"reason":"ok"}'
                            }
                        }
                    ]
                }
            ),
        )
        session = Mock(post=Mock(return_value=response))
        session.trust_env = True

        with patch(
            "custom.btc_agent.llm_decision._get_openai_rest_session",
            return_value=session,
        ), patch(
            "builtins.print",
        ):
            content = _request_openai_once(
                model="gpt-4.1-mini",
                api_key="test-key",
                system_prompt="system",
                user_prompt="user",
                timeout_seconds=15.0,
            )

        self.assertIn('"decision":"UP"', content)
        session.post.assert_called_once()
        _, kwargs = session.post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"]["model"], "gpt-4.1-mini")
        self.assertEqual(kwargs["json"]["messages"][0], {"role": "system", "content": "system"})
        self.assertEqual(kwargs["json"]["messages"][1], {"role": "user", "content": "user"})
        self.assertEqual(kwargs["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["json"]["max_tokens"], 100)
        self.assertEqual(kwargs["timeout"], 15.0)

    def test_get_openai_rest_session_reuses_trust_env_false_session(self):
        from custom.btc_agent import llm_decision as llm_module

        previous_session = llm_module._OPENAI_REST_SESSION
        llm_module._OPENAI_REST_SESSION = None
        try:
            first = _get_openai_rest_session()
            second = _get_openai_rest_session()
        finally:
            llm_module._OPENAI_REST_SESSION = previous_session

        self.assertIs(first, second)
        self.assertFalse(first.trust_env)

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
