import sys
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object, create_connection=object))
sys.modules.setdefault(
    "agents.polymarket.polymarket",
    types.SimpleNamespace(Polymarket=object),
)

from custom.btc_agent.executor import (
    cancel_live_order,
    _extract_minimum_size_from_error,
    _get_order_notional,
    _quantize_live_buy_size_for_amount_precision,
    _scale_live_size_for_min_notional,
    _execute_paper_trade,
    _execute_live_trade,
    _validate_trade_candidate,
    evaluate_ok_to_submit,
    get_effective_decision_confidence,
    get_account_balance_snapshot,
    get_effective_min_confidence,
    get_token_quote_snapshot,
    get_submission_limit_price,
    get_submission_limit_label,
    retry_unfilled_live_order,
    TokenQuoteSnapshot,
)


class TestBtcExecutor(unittest.TestCase):
    def test_get_token_quote_snapshot_populates_book_imbalance_fields(self):
        with patch(
            "custom.btc_agent.executor._get_price_from_clob_single",
            side_effect=lambda token_id, side: 0.50 if side == "BUY" else 0.49,
        ), patch(
            "custom.btc_agent.executor._get_midpoint_price",
            return_value=0.50,
        ), patch(
            "custom.btc_agent.executor._get_last_trade_price",
            return_value=0.50,
        ), patch(
            "custom.btc_agent.executor._get_orderbook",
            return_value={
                "bids": [
                    {"price": "0.49", "asset_size": "100"},
                    {"price": "0.48", "asset_size": "80"},
                    {"price": "0.47", "asset_size": "60"},
                ],
                "asks": [
                    {"price": "0.51", "asset_size": "50"},
                    {"price": "0.52", "asset_size": "40"},
                    {"price": "0.53", "asset_size": "30"},
                ],
                "tick_size": "0.01",
            },
        ), patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(use_recommended_limit=False),
        ):
            snapshot = get_token_quote_snapshot("token-1")

        self.assertEqual(snapshot.best_bid_size, 100.0)
        self.assertEqual(snapshot.best_ask_size, 50.0)
        self.assertEqual(snapshot.best_bid, 0.49)
        self.assertEqual(snapshot.best_ask, 0.50)
        self.assertAlmostEqual(snapshot.spread, 0.01, places=6)
        self.assertAlmostEqual(snapshot.top_level_book_imbalance, 240.0 / 360.0, places=6)
        self.assertAlmostEqual(snapshot.imbalance_pressure, (240.0 - 120.0) / 360.0, places=6)

    def test_get_token_quote_snapshot_prefers_quote_pair_over_book_extremes_for_spread(self):
        with patch(
            "custom.btc_agent.executor._get_price_from_clob_single",
            side_effect=lambda token_id, side: 0.68 if side == "BUY" else 0.67,
        ), patch(
            "custom.btc_agent.executor._get_midpoint_price",
            return_value=0.675,
        ), patch(
            "custom.btc_agent.executor._get_last_trade_price",
            return_value=0.675,
        ), patch(
            "custom.btc_agent.executor._get_orderbook",
            return_value={
                "bids": [{"price": "0.01", "asset_size": "100"}],
                "asks": [{"price": "0.99", "asset_size": "100"}],
                "tick_size": "0.01",
            },
        ), patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(use_recommended_limit=False),
        ):
            snapshot = get_token_quote_snapshot("token-1")

        self.assertEqual(snapshot.best_bid, 0.67)
        self.assertEqual(snapshot.best_ask, 0.68)
        self.assertAlmostEqual(snapshot.spread, 0.01, places=6)

    def test_extract_minimum_size_from_error_parses_exchange_response(self):
        exc = Exception("order abc is invalid. Size (2.88) lower than the minimum: 5")
        self.assertEqual(_extract_minimum_size_from_error(exc), 5.0)

    def test_get_submission_limit_price_prefers_target_when_recommended_disabled(self):
        snapshot = types.SimpleNamespace(
            recommended_limit_price=0.42,
            target_limit_price=0.40,
        )
        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(use_recommended_limit=False),
        ):
            self.assertEqual(get_submission_limit_price(snapshot), 0.40)
            self.assertEqual(get_submission_limit_label(), "target limit")

    def test_evaluate_ok_to_submit_uses_target_limit_label(self):
        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(use_recommended_limit=False),
        ):
            ok, reason = evaluate_ok_to_submit(
                buy_quote=0.41,
                reference_price=0.40,
                submission_limit_price=0.40,
                tick_size=0.01,
            )

        self.assertTrue(ok)
        self.assertIn("target limit", reason)

    def test_scale_live_size_for_min_notional_adds_buffer_above_exchange_minimum(self):
        limit_price = 0.19992

        size = _scale_live_size_for_min_notional(
            base_size=5.0,
            limit_price=limit_price,
            min_order_usd=1.0,
        )
        order_notional = _get_order_notional(size, limit_price)

        self.assertGreaterEqual(order_notional, 1.01)

    def test_quantize_live_buy_size_for_amount_precision_rounds_down_to_valid_quantum(self):
        self.assertEqual(_quantize_live_buy_size_for_amount_precision(0.83, 2.4096), 2.0)
        self.assertEqual(_quantize_live_buy_size_for_amount_precision(0.25, 2.4096), 2.4)

    def test_quantize_live_buy_size_for_amount_precision_keeps_min_positive_quantum(self):
        self.assertGreater(_quantize_live_buy_size_for_amount_precision(0.999, 0.0001), 0.0)

    def test_execute_paper_trade_populates_phase3_execution_metrics(self):
        decision = types.SimpleNamespace(side="UP", confidence=0.8)
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.50,
            midpoint=0.51,
            last_trade_price=0.52,
            reference_price=0.51,
            target_limit_price=0.51,
            recommended_limit_price=0.51,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.50,
            best_ask=0.51,
            tick_size=0.01,
            spread=0.01,
            best_bid_size=20.0,
            best_ask_size=12.5,
        )
        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                use_recommended_limit=False,
                shares_per_trade=5.0,
            ),
        ):
            result = _execute_paper_trade(decision, snapshot, effective_confidence=0.8)

        self.assertTrue(result.executed)
        self.assertEqual(result.quoted_price_at_entry, 0.50)
        self.assertEqual(result.actual_fill_price, 0.51)
        self.assertAlmostEqual(result.realized_slippage_bps, 200.0, places=6)
        self.assertEqual(result.order_latency_ms, 0)
        self.assertEqual(result.book_depth_at_fill, 12.5)
        self.assertEqual(result.shares_requested, result.size)

    def test_account_balance_snapshot_uses_pusd_as_cash_balance(self):
        with patch(
            "custom.btc_agent.executor.get_polymarket_config",
            return_value=types.SimpleNamespace(
                private_key="0xabc",
                proxy_address=None,
                polygon_rpc="https://polygon.drpc.org",
                polygon_rpc_urls=["https://polygon.drpc.org"],
                data_api="https://data-api.polymarket.com",
            ),
        ), patch(
            "custom.btc_agent.executor._derive_signer_address",
            return_value="0x123",
        ), patch(
            "custom.btc_agent.executor._get_polygon_pusd_balance",
            return_value=32.207,
        ), patch(
            "custom.btc_agent.executor._get_polygon_usdc_balance",
            return_value=0.0,
        ), patch(
            "custom.btc_agent.executor._get_portfolio_value",
            return_value=4.5,
        ):
            snapshot = get_account_balance_snapshot()

        self.assertEqual(snapshot.cash_balance, 32.207)
        self.assertEqual(snapshot.legacy_usdc_balance, 0.0)
        self.assertEqual(snapshot.total_account_value, 36.707)

    def test_validate_trade_candidate_uses_edge_not_llm_max_price(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=9999999999,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.90,
            max_price_to_pay=0.20,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.50,
            midpoint=0.50,
            last_trade_price=0.50,
            reference_price=0.50,
            target_limit_price=0.50,
            recommended_limit_price=0.50,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.49,
            best_ask=0.50,
            tick_size=0.01,
            spread=0.01,
        )

        validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIs(validated_snapshot, snapshot)
        self.assertIsNone(rejection)

    def test_validate_trade_candidate_rejects_below_min_confidence(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            start_ts=1_000_000_000,
            end_ts=1_000_000_300,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.48,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.50,
            midpoint=0.50,
            last_trade_price=0.50,
            reference_price=0.50,
            target_limit_price=0.50,
            recommended_limit_price=0.50,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.49,
            best_ask=0.50,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_275, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime, patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                min_confidence=0.70,
                discovery_min_confidence=0.10,
                final_window_min_confidence=0.75,
                disable_liquidity_filter=False,
                use_recommended_limit=False,
            ),
        ):
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIsNone(validated_snapshot)
        self.assertIsNotNone(rejection)
        self.assertIn("Confidence floor veto", rejection.reason)

    def test_validate_trade_candidate_preserves_intent_quote_and_depth_on_rejection(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=9999999999,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.48,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.50,
            midpoint=0.50,
            last_trade_price=0.50,
            reference_price=0.50,
            target_limit_price=0.50,
            recommended_limit_price=0.50,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.49,
            best_ask=0.50,
            tick_size=0.01,
            spread=0.01,
            best_bid_size=8.0,
            best_ask_size=12.5,
        )

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                min_confidence=0.70,
                disable_liquidity_filter=False,
                use_recommended_limit=False,
                paper_trading=True,
                shares_per_trade=5.0,
            ),
        ):
            validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIsNone(validated_snapshot)
        self.assertIsNotNone(rejection)
        self.assertEqual(rejection.quoted_price_at_entry, 0.50)
        self.assertEqual(rejection.book_depth_at_fill, 12.5)
        self.assertGreater(rejection.shares_requested, 0.0)
        self.assertEqual(rejection.order_latency_ms, 0)

    def test_get_effective_min_confidence_uses_default_mid_window(self):
        market = types.SimpleNamespace(
            slug="btc-updown-5m-1000000000",
            start_ts=1_000_000_000,
            end_ts=1_000_000_300,
        )
        features = types.SimpleNamespace(adx_14=42.0)
        fake_now = datetime.fromtimestamp(1_000_000_120, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime, patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                min_confidence=0.65,
                discovery_min_confidence=0.85,
                trend_priority_adx_threshold=30.0,
                trend_relaxed_min_confidence=0.62,
                final_window_min_confidence=0.70,
            ),
        ):
            mock_datetime.now.return_value = fake_now
            self.assertEqual(get_effective_min_confidence(market, features=features), 0.65)

    def test_get_effective_min_confidence_uses_discovery_floor_early(self):
        market = types.SimpleNamespace(
            slug="btc-updown-5m-1000000000",
            start_ts=1_000_000_000,
            end_ts=1_000_000_300,
        )
        features = types.SimpleNamespace(adx_14=42.0)
        fake_now = datetime.fromtimestamp(1_000_000_050, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime, patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                min_confidence=0.65,
                discovery_min_confidence=0.85,
                trend_priority_adx_threshold=30.0,
                trend_relaxed_min_confidence=0.62,
                final_window_min_confidence=0.70,
            ),
        ):
            mock_datetime.now.return_value = fake_now
            self.assertEqual(get_effective_min_confidence(market, features=features), 0.85)

    def test_get_effective_min_confidence_raises_floor_in_last_minute(self):
        market = types.SimpleNamespace(
            slug="btc-updown-5m-1000000000",
            start_ts=1_000_000_000,
            end_ts=1_000_000_300,
        )
        features = types.SimpleNamespace(adx_14=50.0)
        fake_now = datetime.fromtimestamp(1_000_000_245, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime, patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                min_confidence=0.65,
                discovery_min_confidence=0.85,
                trend_priority_adx_threshold=30.0,
                trend_relaxed_min_confidence=0.62,
                final_window_min_confidence=0.70,
            ),
        ):
            mock_datetime.now.return_value = fake_now
            self.assertEqual(get_effective_min_confidence(market, features=features), 0.70)

    def test_get_effective_decision_confidence_boosts_when_already_winning_by_half_atr(self):
        market = types.SimpleNamespace(
            settlement_threshold=100.0,
        )
        decision = types.SimpleNamespace(side="UP", confidence=0.66)
        features = types.SimpleNamespace(price_usd=104.0, atr_14=6.0)

        effective_confidence = get_effective_decision_confidence(
            decision,
            market,
            features=features,
        )

        self.assertAlmostEqual(effective_confidence, 0.81, places=6)

    def test_evaluate_ok_to_submit_allows_four_tick_buffer_in_extreme_volatility(self):
        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(use_recommended_limit=True),
        ):
            ok, reason = evaluate_ok_to_submit(
                buy_quote=0.539,
                reference_price=0.50,
                submission_limit_price=0.50,
                tick_size=0.01,
                volatility_5m=30.0,
            )

        self.assertTrue(ok)
        self.assertIn("within 4 ticks", reason)

    def test_validate_trade_candidate_rejects_large_consensus_gap(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            up_market_probability=0.95,
            down_market_probability=0.02,
            end_ts=9999999999,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="DOWN",
            confidence=0.90,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="down-token",
            buy_quote=0.20,
            midpoint=0.20,
            last_trade_price=0.20,
            reference_price=0.20,
            target_limit_price=0.20,
            recommended_limit_price=0.20,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.19,
            best_ask=0.20,
            tick_size=0.01,
            spread=0.01,
        )

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                min_confidence=0.70,
                disable_liquidity_filter=False,
                use_recommended_limit=False,
            ),
        ):
            validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIsNone(validated_snapshot)
        self.assertIsNotNone(rejection)
        self.assertIn("Consensus-gap veto", rejection.reason)

    def test_validate_trade_candidate_allows_high_confidence_trade_with_zero_edge_buffer(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=9999999999,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.92,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.83,
            midpoint=0.83,
            last_trade_price=0.83,
            reference_price=0.83,
            target_limit_price=0.83,
            recommended_limit_price=0.83,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.82,
            best_ask=0.83,
            tick_size=0.01,
            spread=0.01,
        )

        validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIs(validated_snapshot, snapshot)
        self.assertIsNone(rejection)

    def test_validate_trade_candidate_rejects_sub_015_quote_before_final_15_seconds(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=1_000_000_100,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="DOWN",
            confidence=0.90,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="down-token",
            buy_quote=0.01,
            midpoint=0.01,
            last_trade_price=0.01,
            reference_price=0.01,
            target_limit_price=0.01,
            recommended_limit_price=0.01,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.01,
            best_ask=0.02,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIsNone(validated_snapshot)
        self.assertIsNotNone(rejection)
        self.assertIn("Quote-floor veto", rejection.reason)

    def test_validate_trade_candidate_allows_sub_010_quote_in_discovery_phase(self):
        market = types.SimpleNamespace(
            slug="btc-updown-5m-1000000000",
            up_token_id="up-token",
            down_token_id="down-token",
            up_market_probability=0.20,
            down_market_probability=0.80,
            end_ts=1_000_000_300,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="DOWN",
            confidence=0.90,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="down-token",
            buy_quote=0.09,
            midpoint=0.09,
            last_trade_price=0.09,
            reference_price=0.09,
            target_limit_price=0.09,
            recommended_limit_price=0.09,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.08,
            best_ask=0.09,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_050, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIs(validated_snapshot, snapshot)
        self.assertIsNone(rejection)

    def test_validate_trade_candidate_allows_sub_015_quote_in_mid_window_band(self):
        market = types.SimpleNamespace(
            slug="btc-updown-5m-1000000000",
            up_token_id="up-token",
            down_token_id="down-token",
            up_market_probability=0.28,
            down_market_probability=0.72,
            end_ts=1_000_000_300,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="DOWN",
            confidence=0.80,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="down-token",
            buy_quote=0.12,
            midpoint=0.12,
            last_trade_price=0.12,
            reference_price=0.12,
            target_limit_price=0.12,
            recommended_limit_price=0.12,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.11,
            best_ask=0.12,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_150, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIs(validated_snapshot, snapshot)
        self.assertIsNone(rejection)

    def test_validate_trade_candidate_rejects_down_when_rsi9_is_oversold(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            settlement_threshold=100.0,
            end_ts=1_000_000_180,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="DOWN",
            confidence=0.80,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(
            price_usd=95.0,
            volatility_5m=10.0,
            rsi_9=27.0,
            delta_pct_from_window_open=-0.001,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="down-token",
            buy_quote=0.30,
            midpoint=0.30,
            last_trade_price=0.30,
            reference_price=0.30,
            target_limit_price=0.30,
            recommended_limit_price=0.30,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.29,
            best_ask=0.30,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(
                market, decision, features=features, snapshot=snapshot
            )

        self.assertIsNone(validated_snapshot)
        self.assertIn("RSI directional veto blocked DOWN", rejection.reason)

    def test_validate_trade_candidate_rejects_down_on_hard_exhaustion_cap(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            settlement_threshold=100.0,
            end_ts=1_000_000_180,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="DOWN",
            confidence=0.95,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(
            price_usd=95.0,
            volatility_5m=10.0,
            rsi_9=24.0,
            adx_14=55.0,
            delta_pct_from_window_open=-0.001,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="down-token",
            buy_quote=0.30,
            midpoint=0.30,
            last_trade_price=0.30,
            reference_price=0.30,
            target_limit_price=0.30,
            recommended_limit_price=0.30,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.29,
            best_ask=0.30,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(
                market, decision, features=features, snapshot=snapshot
            )

        self.assertIsNone(validated_snapshot)
        self.assertIn("Hard Exhaustion Veto: ADX > 50 and RSI9 < 25", rejection.reason)

    def test_validate_trade_candidate_rejects_up_when_rsi9_is_overbought(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            settlement_threshold=100.0,
            end_ts=1_000_000_180,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.80,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(
            price_usd=105.0,
            volatility_5m=10.0,
            rsi_9=72.0,
            delta_pct_from_window_open=0.001,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.60,
            midpoint=0.60,
            last_trade_price=0.60,
            reference_price=0.60,
            target_limit_price=0.60,
            recommended_limit_price=0.60,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.59,
            best_ask=0.60,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(
                market, decision, features=features, snapshot=snapshot
            )

        self.assertIsNone(validated_snapshot)
        self.assertIn("RSI directional veto blocked UP", rejection.reason)

    def test_validate_trade_candidate_rejects_up_on_hard_exhaustion_cap(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            settlement_threshold=100.0,
            end_ts=1_000_000_180,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.95,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(
            price_usd=105.0,
            volatility_5m=10.0,
            rsi_9=82.0,
            adx_14=55.0,
            delta_pct_from_window_open=0.001,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.60,
            midpoint=0.60,
            last_trade_price=0.60,
            reference_price=0.60,
            target_limit_price=0.60,
            recommended_limit_price=0.60,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.59,
            best_ask=0.60,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(
                market, decision, features=features, snapshot=snapshot
            )

        self.assertIsNone(validated_snapshot)
        self.assertIn("Hard Exhaustion Veto: ADX > 50 and RSI9 > 75", rejection.reason)

    def test_validate_trade_candidate_rejects_up_on_absolute_rsi_cap(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            settlement_threshold=100.0,
            end_ts=1_000_000_180,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.95,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(
            price_usd=105.0,
            volatility_5m=10.0,
            rsi_9=86.0,
            adx_14=30.0,
            delta_pct_from_window_open=0.001,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.60,
            midpoint=0.60,
            last_trade_price=0.60,
            reference_price=0.60,
            target_limit_price=0.60,
            recommended_limit_price=0.60,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.59,
            best_ask=0.60,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(
                market, decision, features=features, snapshot=snapshot
            )

        self.assertIsNone(validated_snapshot)
        self.assertIn("Absolute RSI Veto blocked UP", rejection.reason)
        self.assertIn("threshold=85.000", rejection.reason)

    def test_validate_trade_candidate_rejects_down_on_absolute_rsi_cap(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            settlement_threshold=100.0,
            end_ts=1_000_000_180,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="DOWN",
            confidence=0.95,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(
            price_usd=95.0,
            volatility_5m=10.0,
            rsi_9=14.0,
            adx_14=30.0,
            delta_pct_from_window_open=-0.001,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="down-token",
            buy_quote=0.60,
            midpoint=0.60,
            last_trade_price=0.60,
            reference_price=0.60,
            target_limit_price=0.60,
            recommended_limit_price=0.60,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.59,
            best_ask=0.60,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(
                market, decision, features=features, snapshot=snapshot
            )

        self.assertIsNone(validated_snapshot)
        self.assertIn("Absolute RSI Veto blocked DOWN", rejection.reason)
        self.assertIn("threshold=15.000", rejection.reason)

    def test_validate_trade_candidate_allows_up_when_parabolic_rsi_suspension_applies(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            settlement_threshold=100.0,
            end_ts=1_000_000_180,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.80,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(
            price_usd=105.0,
            volatility_5m=10.0,
            rsi_9=84.0,
            rsi_speed_divergence=6.0,
            adx_14=40.0,
            delta_pct_from_window_open=0.001,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.60,
            midpoint=0.60,
            last_trade_price=0.60,
            reference_price=0.60,
            target_limit_price=0.60,
            recommended_limit_price=0.60,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.59,
            best_ask=0.60,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(
                market, decision, features=features, snapshot=snapshot
            )

        self.assertIs(validated_snapshot, snapshot)
        self.assertIsNone(rejection)

    def test_validate_trade_candidate_rejects_late_itm_entry_when_quote_is_too_high(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            settlement_threshold=100.0,
            end_ts=1_000_000_050,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="DOWN",
            confidence=0.95,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(
            price_usd=95.0,
            volatility_5m=10.0,
            rsi_9=35.0,
            adx_14=35.0,
            delta_pct_from_window_open=-0.001,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="down-token",
            buy_quote=0.90,
            midpoint=0.90,
            last_trade_price=0.90,
            reference_price=0.90,
            target_limit_price=0.90,
            recommended_limit_price=0.90,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.89,
            best_ask=0.90,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(
                market, decision, features=features, snapshot=snapshot
            )

        self.assertIsNone(validated_snapshot)
        self.assertIn("Late ITM entry quote veto", rejection.reason)

    def test_validate_trade_candidate_rejects_sub_015_quote_inside_final_15_seconds_when_below_010(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=1_000_000_010,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="DOWN",
            confidence=0.96,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="down-token",
            buy_quote=0.01,
            midpoint=0.01,
            last_trade_price=0.01,
            reference_price=0.01,
            target_limit_price=0.01,
            recommended_limit_price=0.01,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.01,
            best_ask=0.02,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIsNone(validated_snapshot)
        self.assertIsNotNone(rejection)
        self.assertIn("Consensus-gap veto", rejection.reason)

    def test_validate_trade_candidate_rejects_sub_010_quote_inside_final_minute(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=1_000_000_050,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="DOWN",
            confidence=0.95,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="down-token",
            buy_quote=0.02,
            midpoint=0.02,
            last_trade_price=0.02,
            reference_price=0.02,
            target_limit_price=0.02,
            recommended_limit_price=0.02,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.01,
            best_ask=0.02,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIsNone(validated_snapshot)
        self.assertIsNotNone(rejection)
        self.assertIn("Quote-floor veto", rejection.reason)

    def test_validate_trade_candidate_rejects_required_velocity_above_volatility_threshold(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            settlement_threshold=100.0,
            end_ts=1_000_000_020,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="DOWN",
            confidence=0.85,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(
            price_usd=180.0,
            volatility_5m=10.0,
            rsi_9=40.0,
            delta_pct_from_window_open=-0.001,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="down-token",
            buy_quote=0.80,
            midpoint=0.80,
            last_trade_price=0.80,
            reference_price=0.80,
            target_limit_price=0.80,
            recommended_limit_price=0.80,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.79,
            best_ask=0.80,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(
                market, decision, features=features, snapshot=snapshot
            )

        self.assertIsNone(validated_snapshot)
        self.assertIsNotNone(rejection)
        self.assertIn("Velocity/volatility veto", rejection.reason)

    def test_validate_trade_candidate_rejects_too_close_to_call_margin(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            settlement_threshold=100.0,
            end_ts=1_000_000_180,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="DOWN",
            confidence=0.75,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(
            price_usd=100.5,
            volatility_5m=10.0,
            rsi_9=50.0,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="down-token",
            buy_quote=0.50,
            midpoint=0.50,
            last_trade_price=0.50,
            reference_price=0.50,
            target_limit_price=0.50,
            recommended_limit_price=0.50,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.49,
            best_ask=0.50,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(
                market, decision, features=features, snapshot=snapshot
            )

        self.assertIsNone(validated_snapshot)
        self.assertIn("Too close to call", rejection.reason)

    def test_validate_trade_candidate_rejects_up_quote_price_divergence(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            settlement_threshold=100.0,
            end_ts=1_000_000_180,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.80,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(
            price_usd=112.0,
            volatility_5m=20.0,
            rsi_9=60.0,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.36,
            midpoint=0.36,
            last_trade_price=0.36,
            reference_price=0.36,
            target_limit_price=0.36,
            recommended_limit_price=0.36,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.35,
            best_ask=0.36,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(
                market, decision, features=features, snapshot=snapshot
            )

        self.assertIsNone(validated_snapshot)
        self.assertIn("Quote-price divergence veto", rejection.reason)

    def test_validate_trade_candidate_rejects_up_below_strike_when_rsi_hot(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            settlement_threshold=100.0,
            end_ts=1_000_000_180,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.80,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(
            price_usd=98.0,
            volatility_5m=20.0,
            rsi_9=61.0,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.46,
            midpoint=0.46,
            last_trade_price=0.46,
            reference_price=0.46,
            target_limit_price=0.46,
            recommended_limit_price=0.46,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.45,
            best_ask=0.46,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(
                market, decision, features=features, snapshot=snapshot
            )

        self.assertIsNone(validated_snapshot)
        self.assertIn("Momentum-trap veto blocked UP trade below the strike", rejection.reason)

    def test_validate_trade_candidate_rejects_up_rsi_ceiling_above_strike(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            settlement_threshold=100.0,
            end_ts=1_000_000_050,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.95,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(
            price_usd=101.0,
            volatility_5m=2.0,
            rsi_9=90.0,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.80,
            midpoint=0.80,
            last_trade_price=0.80,
            reference_price=0.80,
            target_limit_price=0.80,
            recommended_limit_price=0.80,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.79,
            best_ask=0.80,
            tick_size=0.01,
            spread=0.01,
        )

        fake_now = datetime.fromtimestamp(1_000_000_000, tz=timezone.utc)
        with patch("custom.btc_agent.executor.datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            validated_snapshot, rejection = _validate_trade_candidate(
                market, decision, features=features, snapshot=snapshot
            )

        self.assertIsNone(validated_snapshot)
        self.assertIn("Absolute RSI Veto blocked UP", rejection.reason)

    def test_execute_paper_trade_uses_fixed_shares_per_trade(self):
        from custom.btc_agent.executor import _execute_paper_trade

        decision = types.SimpleNamespace(side="UP", confidence=0.91, max_price_to_pay=1.0)
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.90,
            midpoint=0.90,
            last_trade_price=0.90,
            reference_price=0.90,
            target_limit_price=0.90,
            recommended_limit_price=0.90,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.89,
            best_ask=0.90,
            tick_size=0.01,
            spread=0.01,
        )

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                shares_per_trade=5.0,
                use_recommended_limit=False,
            ),
        ):
            result = _execute_paper_trade(decision=decision, snapshot=snapshot)

        self.assertTrue(result.executed)
        self.assertEqual(result.size, 5.0)
        self.assertIn("shares_per_trade=5.0000", result.reason)

    def test_validate_trade_candidate_allows_t5_deadline_execution_despite_negative_edge(self):
        fake_now_ts = int(datetime.now(timezone.utc).timestamp())
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=fake_now_ts + 4,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.86,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.89,
            midpoint=0.89,
            last_trade_price=0.89,
            reference_price=0.89,
            target_limit_price=0.89,
            recommended_limit_price=0.89,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.88,
            best_ask=0.89,
            tick_size=0.01,
            spread=0.01,
        )

        validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIs(validated_snapshot, snapshot)
        self.assertIsNone(rejection)

    def test_validate_trade_candidate_allows_window_delta_master_switch(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=int(datetime.now(timezone.utc).timestamp()) + 8,
            volume=100.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.90,
            max_price_to_pay=1.0,
            reason="test",
        )
        features = types.SimpleNamespace(delta_pct_from_window_open=0.0016)
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.90,
            midpoint=0.90,
            last_trade_price=0.90,
            reference_price=0.90,
            target_limit_price=0.90,
            recommended_limit_price=0.90,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.89,
            best_ask=0.90,
            tick_size=0.01,
            spread=0.01,
        )

        validated_snapshot, rejection = _validate_trade_candidate(
            market,
            decision,
            features=features,
            snapshot=snapshot,
        )

        self.assertIs(validated_snapshot, snapshot)
        self.assertIsNone(rejection)

    def test_validate_trade_candidate_blocks_high_price_low_liquidity_trade(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=0,
            volume=500.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.97,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.90,
            midpoint=0.90,
            last_trade_price=0.90,
            reference_price=0.90,
            target_limit_price=0.90,
            recommended_limit_price=0.90,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.89,
            best_ask=0.90,
            tick_size=0.01,
            spread=0.01,
        )

        validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIsNone(validated_snapshot)
        self.assertIsNotNone(rejection)
        self.assertIn("liquidity filter", rejection.reason)

    def test_validate_trade_candidate_blocks_thin_liquidity_by_spread(self):
        fake_now_ts = int(datetime.now(timezone.utc).timestamp())
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=fake_now_ts + 180,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.90,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.82,
            midpoint=0.82,
            last_trade_price=0.82,
            reference_price=0.82,
            target_limit_price=0.82,
            recommended_limit_price=0.82,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.795,
            best_ask=0.845,
            tick_size=0.01,
            spread=0.05,
            spread_bps=2926.8,
        )

        validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIsNone(validated_snapshot)
        self.assertIsNotNone(rejection)
        self.assertIn("Thin liquidity blocked execution", rejection.reason)

    def test_validate_trade_candidate_blocks_spread_above_configured_max(self):
        fake_now_ts = int(datetime.now(timezone.utc).timestamp())
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=fake_now_ts + 180,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.90,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.82,
            midpoint=0.82,
            last_trade_price=0.82,
            reference_price=0.82,
            target_limit_price=0.82,
            recommended_limit_price=0.82,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.70,
            best_ask=0.82,
            tick_size=0.01,
            spread=0.12,
            spread_bps=1400.0,
        )

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                min_confidence=0.65,
                discovery_min_confidence=0.65,
                trend_priority_adx_threshold=30.0,
                trend_relaxed_min_confidence=0.62,
                final_window_min_confidence=0.85,
                min_execution_edge=0.02,
                max_spread=0.10,
                use_recommended_limit=True,
                disable_liquidity_filter=True,
                market_win_chance_veto_threshold=0.15,
                market_win_chance_veto_end_seconds=120,
                down_rsi_veto_threshold=30.0,
                up_rsi_veto_base_threshold=70.0,
                up_rsi_veto_trend_threshold=85.0,
                up_rsi_veto_adx_threshold=30.0,
                parabolic_rsi_speed_divergence_threshold=5.0,
                parabolic_rsi_suspend_adx_threshold=35.0,
                required_velocity_divisor=5.0,
                itm_confidence_boost_usd=20.0,
                itm_confidence_boost_atr_multiplier=0.50,
                itm_confidence_boost_amount=0.15,
            ),
        ):
            validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIsNone(validated_snapshot)
        self.assertIsNotNone(rejection)
        self.assertIn("Spread veto blocked execution", rejection.reason)

    def test_validate_trade_candidate_allows_high_price_low_liquidity_trade_when_filter_disabled(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=0,
            volume=500.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.97,
            max_price_to_pay=1.0,
            reason="test",
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.90,
            midpoint=0.90,
            last_trade_price=0.90,
            reference_price=0.90,
            target_limit_price=0.90,
            recommended_limit_price=0.90,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.89,
            best_ask=0.90,
            tick_size=0.01,
            spread=0.01,
        )

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                use_recommended_limit=False,
                disable_liquidity_filter=True,
            ),
        ):
            validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIs(validated_snapshot, snapshot)
        self.assertIsNone(rejection)

    def test_execute_live_trade_retries_gtc_after_fok_full_fill_error(self):
        market = types.SimpleNamespace(end_ts=int(datetime.now(timezone.utc).timestamp()) + 8)
        decision = types.SimpleNamespace(side="UP", confidence=0.8, max_price_to_pay=1.0)
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.60,
            midpoint=0.60,
            last_trade_price=0.60,
            reference_price=0.60,
            target_limit_price=0.60,
            recommended_limit_price=0.60,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.59,
            best_ask=0.60,
            tick_size=0.01,
            spread=0.01,
        )
        client = types.SimpleNamespace(
            execute_order=unittest.mock.Mock(
                side_effect=[
                    Exception("order couldn't be fully filled. FOK orders are fully filled or killed."),
                    {"ok": True},
                ]
            )
        )

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                shares_per_trade=5.0,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
            ),
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
        ), patch(
            "custom.btc_agent.executor._resolve_actual_fill_price",
            return_value=0.70,
        ):
            result = _execute_live_trade(decision=decision, market=market, snapshot=snapshot)

        self.assertTrue(result.executed)
        self.assertIn("GTC (after FOK retry)", result.reason)
        self.assertEqual(client.execute_order.call_count, 2)
        self.assertTrue(client.execute_order.call_args_list[0].kwargs["use_fok"])
        self.assertFalse(client.execute_order.call_args_list[1].kwargs["use_fok"])

    def test_execute_live_trade_returns_clean_rejection_for_final_deadline_fok_failure(self):
        market = types.SimpleNamespace(end_ts=0)
        decision = types.SimpleNamespace(side="UP", confidence=0.8, max_price_to_pay=1.0)
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.60,
            midpoint=0.60,
            last_trade_price=0.60,
            reference_price=0.60,
            target_limit_price=0.60,
            recommended_limit_price=0.60,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.59,
            best_ask=0.60,
            tick_size=0.01,
            spread=0.01,
        )
        client = types.SimpleNamespace(
            execute_order=unittest.mock.Mock(
                side_effect=Exception(
                    "order couldn't be fully filled. FOK orders are fully filled or killed."
                )
            )
        )

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                shares_per_trade=5.0,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
            ),
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
        ), patch(
            "custom.btc_agent.executor._resolve_actual_fill_price",
            return_value=0.70,
        ):
            result = _execute_live_trade(decision=decision, market=market, snapshot=snapshot)

        self.assertFalse(result.executed)
        self.assertIn("FOK order could not be fully filled", result.reason)
        self.assertEqual(client.execute_order.call_count, 1)

    def test_execute_live_trade_returns_rejection_when_minimum_size_exceeds_shares_per_trade(self):
        market = types.SimpleNamespace(end_ts=int(datetime.now(timezone.utc).timestamp()) + 30)
        decision = types.SimpleNamespace(side="UP", confidence=0.8, max_price_to_pay=1.0)
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.70,
            midpoint=0.70,
            last_trade_price=0.70,
            reference_price=0.70,
            target_limit_price=0.70,
            recommended_limit_price=0.70,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.69,
            best_ask=0.70,
            tick_size=0.01,
            spread=0.01,
        )
        client = types.SimpleNamespace(
            execute_order=unittest.mock.Mock(
                side_effect=Exception("order xyz is invalid. Size (2.88) lower than the minimum: 5")
            )
        )

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                shares_per_trade=2.0,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
            ),
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
        ), patch(
            "custom.btc_agent.executor._resolve_actual_fill_price",
            return_value=0.70,
        ):
            result = _execute_live_trade(decision=decision, market=market, snapshot=snapshot)

        self.assertFalse(result.executed)
        self.assertIn("Exchange minimum size exceeds configured shares_per_trade", result.reason)
        self.assertEqual(client.execute_order.call_count, 1)

    def test_execute_live_trade_uses_fixed_shares_per_trade(self):
        market = types.SimpleNamespace(end_ts=int(datetime.now(timezone.utc).timestamp()) + 30)
        decision = types.SimpleNamespace(side="UP", confidence=0.95, max_price_to_pay=1.0)
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.70,
            midpoint=0.70,
            last_trade_price=0.70,
            reference_price=0.70,
            target_limit_price=0.70,
            recommended_limit_price=0.70,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.69,
            best_ask=0.70,
            tick_size=0.01,
            spread=0.01,
        )
        client = types.SimpleNamespace(execute_order=unittest.mock.Mock(return_value={"ok": True}))

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                shares_per_trade=5.0,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
            ),
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
        ), patch(
            "custom.btc_agent.executor._resolve_actual_fill_price",
            return_value=0.70,
        ):
            result = _execute_live_trade(decision=decision, market=market, snapshot=snapshot)

        self.assertTrue(result.executed)
        self.assertEqual(result.size, 5.0)
        self.assertIn("for 5.0000 shares", result.reason)
        self.assertEqual(client.execute_order.call_args.kwargs["size"], 5.0)

    def test_execute_live_trade_triggers_slippage_kill_switch_after_bad_fill(self):
        market = types.SimpleNamespace(end_ts=int(datetime.now(timezone.utc).timestamp()) + 30)
        decision = types.SimpleNamespace(side="UP", confidence=0.95, max_price_to_pay=1.0)
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.70,
            midpoint=0.70,
            last_trade_price=0.70,
            reference_price=0.70,
            target_limit_price=0.70,
            recommended_limit_price=0.70,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.69,
            best_ask=0.70,
            tick_size=0.01,
            spread=0.01,
        )
        client = types.SimpleNamespace(execute_order=unittest.mock.Mock(return_value={"ok": True}))

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                shares_per_trade=5.0,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
                slippage_cooldown_threshold_bps=500.0,
            ),
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
        ), patch(
            "custom.btc_agent.executor._resolve_actual_fill_price",
            return_value=0.75,
        ), patch(
            "custom.btc_agent.executor.set_trade_cooldown",
        ) as mock_set_cooldown:
            result = _execute_live_trade(decision=decision, market=market, snapshot=snapshot)

        self.assertTrue(result.executed)
        mock_set_cooldown.assert_called_once_with(5)
        self.assertIn("slippage kill-switch triggered", result.reason)
        self.assertIn("threshold_bps=500.000", result.reason)

    def test_execute_live_trade_quantizes_size_for_market_buy_precision(self):
        market = types.SimpleNamespace(end_ts=int(datetime.now(timezone.utc).timestamp()) + 30)
        decision = types.SimpleNamespace(side="UP", confidence=0.80, max_price_to_pay=1.0)
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.83,
            midpoint=0.83,
            last_trade_price=0.83,
            reference_price=0.83,
            target_limit_price=0.83,
            recommended_limit_price=0.83,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.82,
            best_ask=0.83,
            tick_size=0.01,
            spread=0.01,
        )
        client = types.SimpleNamespace(execute_order=unittest.mock.Mock(return_value={"ok": True}))

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                shares_per_trade=2.4096,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
            ),
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
        ), patch(
            "custom.btc_agent.executor._resolve_actual_fill_price",
            return_value=0.83,
        ):
            result = _execute_live_trade(decision=decision, market=market, snapshot=snapshot)

        self.assertTrue(result.executed)
        self.assertEqual(result.size, 2.0)
        self.assertEqual(client.execute_order.call_args.kwargs["size"], 2.0)

    def test_execute_live_trade_returns_unfilled_submission_when_fill_not_confirmed(self):
        market = types.SimpleNamespace(end_ts=int(datetime.now(timezone.utc).timestamp()) + 30)
        decision = types.SimpleNamespace(side="UP", confidence=0.80, max_price_to_pay=1.0)
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.67,
            midpoint=0.67,
            last_trade_price=0.67,
            reference_price=0.67,
            target_limit_price=0.67,
            recommended_limit_price=0.67,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.66,
            best_ask=0.67,
            tick_size=0.01,
            spread=0.01,
        )
        client = types.SimpleNamespace(execute_order=unittest.mock.Mock(return_value={"ok": True, "orderID": "abc"}))

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                shares_per_trade=5.0,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
            ),
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
        ), patch(
            "custom.btc_agent.executor._resolve_actual_fill_price",
            return_value=None,
        ):
            result = _execute_live_trade(decision=decision, market=market, snapshot=snapshot)

        self.assertFalse(result.executed)
        self.assertTrue(result.submission_accepted)
        self.assertIsNone(result.actual_fill_price)
        self.assertIn("no fill was confirmed", result.reason)

    def test_execute_live_trade_does_not_treat_generic_response_price_as_fill(self):
        market = types.SimpleNamespace(end_ts=int(datetime.now(timezone.utc).timestamp()) + 30)
        decision = types.SimpleNamespace(side="DOWN", confidence=0.85, max_price_to_pay=1.0)
        snapshot = TokenQuoteSnapshot(
            token_id="down-token",
            buy_quote=0.79,
            midpoint=0.79,
            last_trade_price=0.79,
            reference_price=0.79,
            target_limit_price=0.79,
            recommended_limit_price=0.79,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.78,
            best_ask=0.79,
            tick_size=0.01,
            spread=0.01,
        )
        client = types.SimpleNamespace(
            execute_order=unittest.mock.Mock(return_value={"ok": True, "orderID": "abc", "price": "0.311"})
        )

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                shares_per_trade=5.0,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
                live_order_status_poll_attempts=1,
                live_order_status_poll_interval_seconds=0.0,
            ),
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
        ), patch(
            "custom.btc_agent.executor._fetch_live_order_status",
            return_value={"status": "live", "price": "0.311"},
        ), patch(
            "custom.btc_agent.executor._fetch_actual_fill_price_from_trades",
            return_value=None,
        ):
            result = _execute_live_trade(decision=decision, market=market, snapshot=snapshot)

        self.assertFalse(result.executed)
        self.assertTrue(result.submission_accepted)
        self.assertIsNone(result.actual_fill_price)

    def test_cancel_live_order_uses_client_cancel_order_when_available(self):
        client = types.SimpleNamespace(cancel_order=unittest.mock.Mock(return_value={"ok": True}))
        with patch("custom.btc_agent.executor.Polymarket", return_value=types.SimpleNamespace(client=client)):
            self.assertTrue(cancel_live_order("order-1"))
        client.cancel_order.assert_called_once_with("order-1")

    def test_retry_unfilled_live_order_cancels_and_resubmits_with_latest_snapshot(self):
        market = types.SimpleNamespace(end_ts=int(datetime.now(timezone.utc).timestamp()) + 30)
        order = types.SimpleNamespace(
            side="UP",
            token_id="up-token",
            live_order_id="old-order",
            actual_fill_price=None,
            shares=3.0,
            shares_requested=3.0,
            quoted_price_at_entry=0.67,
            book_depth_at_fill=100.0,
        )
        snapshot = TokenQuoteSnapshot(
            token_id="up-token",
            buy_quote=0.69,
            midpoint=0.69,
            last_trade_price=0.69,
            reference_price=0.69,
            target_limit_price=0.69,
            recommended_limit_price=0.69,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.68,
            best_ask=0.69,
            tick_size=0.01,
            spread=0.01,
        )
        client = types.SimpleNamespace(execute_order=unittest.mock.Mock(return_value={"ok": True, "orderID": "new-order"}))

        with patch(
            "custom.btc_agent.executor.get_trading_config",
            return_value=types.SimpleNamespace(
                paper_trading=False,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
            ),
        ), patch(
            "custom.btc_agent.executor.cancel_live_order",
            return_value=True,
        ), patch(
            "custom.btc_agent.executor.get_token_quote_snapshot",
            return_value=snapshot,
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
        ), patch(
            "custom.btc_agent.executor._resolve_actual_fill_price",
            return_value=0.69,
        ):
            result = retry_unfilled_live_order(order, market)

        self.assertTrue(result.executed)
        self.assertEqual(result.live_order_id, "new-order")
        self.assertEqual(client.execute_order.call_args.kwargs["price"], 0.69)
        self.assertEqual(client.execute_order.call_args.kwargs["size"], 3.0)


if __name__ == "__main__":
    unittest.main()
