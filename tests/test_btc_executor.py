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
    _extract_minimum_size_from_error,
    _get_order_notional,
    _quantize_live_buy_size_for_amount_precision,
    _scale_live_size_for_min_notional,
    _execute_live_trade,
    _validate_trade_candidate,
    evaluate_ok_to_submit,
    get_account_balance_snapshot,
    get_token_quote_snapshot,
    get_submission_limit_price,
    get_submission_limit_label,
    TokenQuoteSnapshot,
)


class TestBtcExecutor(unittest.TestCase):
    def test_get_token_quote_snapshot_populates_book_imbalance_fields(self):
        with patch(
            "custom.btc_agent.executor._get_price_from_clob_single",
            return_value=0.50,
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
        self.assertAlmostEqual(snapshot.top_level_book_imbalance, 240.0 / 360.0, places=6)
        self.assertAlmostEqual(snapshot.imbalance_pressure, (240.0 - 120.0) / 360.0, places=6)

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
            confidence=0.80,
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
            confidence=0.75,
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

    def test_validate_trade_candidate_allows_sub_015_quote_inside_final_15_seconds(self):
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

        self.assertIs(validated_snapshot, snapshot)
        self.assertIsNone(rejection)

    def test_execute_paper_trade_uses_high_confidence_share_override(self):
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
                max_order_price_usd=2.0,
                max_size_high_confidence_threshold=0.9,
                max_size_high_confidence_shares=5.0,
                use_recommended_limit=False,
            ),
        ):
            result = _execute_paper_trade(decision=decision, snapshot=snapshot)

        self.assertTrue(result.executed)
        self.assertEqual(result.size, 5.0)
        self.assertIn("high_confidence_size_override=True", result.reason)

    def test_validate_trade_candidate_allows_t5_deadline_execution_despite_negative_edge(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=0,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.71,
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

    def test_validate_trade_candidate_allows_window_delta_master_switch(self):
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=int(datetime.now(timezone.utc).timestamp()) + 8,
            volume=100.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.60,
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
        market = types.SimpleNamespace(
            up_token_id="up-token",
            down_token_id="down-token",
            end_ts=0,
            volume=5000.0,
        )
        decision = types.SimpleNamespace(
            side="UP",
            confidence=0.80,
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
            best_ask=0.94,
            tick_size=0.01,
            spread=0.24,
            spread_bps=2926.8,
        )

        validated_snapshot, rejection = _validate_trade_candidate(market, decision, snapshot=snapshot)

        self.assertIsNone(validated_snapshot)
        self.assertIsNotNone(rejection)
        self.assertIn("Thin liquidity blocked execution", rejection.reason)

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
                max_order_price_usd=5.0,
                max_size_high_confidence_threshold=1.1,
                max_size_high_confidence_shares=0.0,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
            ),
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
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
                max_order_price_usd=5.0,
                max_size_high_confidence_threshold=1.1,
                max_size_high_confidence_shares=0.0,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
            ),
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
        ):
            result = _execute_live_trade(decision=decision, market=market, snapshot=snapshot)

        self.assertFalse(result.executed)
        self.assertIn("FOK order could not be fully filled", result.reason)
        self.assertEqual(client.execute_order.call_count, 1)

    def test_execute_live_trade_returns_budget_rejection_when_minimum_size_exceeds_budget(self):
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
                max_order_price_usd=2.0,
                max_size_high_confidence_threshold=1.1,
                max_size_high_confidence_shares=0.0,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
            ),
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
        ):
            result = _execute_live_trade(decision=decision, market=market, snapshot=snapshot)

        self.assertFalse(result.executed)
        self.assertIn("Exchange minimum size exceeds configured order budget", result.reason)
        self.assertEqual(client.execute_order.call_count, 1)

    def test_execute_live_trade_uses_high_confidence_share_override(self):
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
                max_order_price_usd=2.0,
                max_size_high_confidence_threshold=0.9,
                max_size_high_confidence_shares=5.0,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
            ),
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
        ):
            result = _execute_live_trade(decision=decision, market=market, snapshot=snapshot)

        self.assertTrue(result.executed)
        self.assertEqual(result.size, 5.0)
        self.assertIn("high_confidence_size_override=True", result.reason)
        self.assertEqual(client.execute_order.call_args.kwargs["size"], 5.0)

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
                max_order_price_usd=2.0,
                max_size_high_confidence_threshold=1.1,
                max_size_high_confidence_shares=0.0,
                live_min_order_usd=1.0,
                live_fee_rate_bps=1000,
                use_recommended_limit=False,
            ),
        ), patch(
            "custom.btc_agent.executor.ensure_live_trade_cash_available",
        ), patch(
            "custom.btc_agent.executor.Polymarket",
            return_value=client,
        ):
            result = _execute_live_trade(decision=decision, market=market, snapshot=snapshot)

        self.assertTrue(result.executed)
        self.assertEqual(result.size, 2.0)
        self.assertEqual(client.execute_order.call_args.kwargs["size"], 2.0)


if __name__ == "__main__":
    unittest.main()
