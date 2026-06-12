import sys
import types
import unittest
import os
import glob
import io
import json
import requests
from datetime import datetime, timezone, timedelta
from contextlib import ExitStack
from unittest.mock import patch
from types import SimpleNamespace

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object, create_connection=object))
sys.modules.setdefault(
    "agents.polymarket.polymarket",
    types.SimpleNamespace(Polymarket=object),
)

from custom.btc_agent.main import (
    _SESSION_LOSS_TRADES,
    _build_regime_fingerprint,
    _should_log_failed_order_attempt,
    append_completed_order_tick,
    append_failed_order_attempt,
    append_failed_live_order,
    append_pending_period_tick_analysis,
    append_period_analysis_sample,
    finalize_current_period_logs_on_exit,
    finalize_pending_period_log,
    enforce_session_loss_trade_limit,
    has_valid_price_to_beat,
    promote_pending_period_log_to_completed,
    run_once,
    main,
    print_features,
    print_llm_decision,
    resolve_price_to_beat_with_retries,
    wait_for_next_tick_or_quit,
    clear_price_to_beat_debug_files,
    write_price_to_beat_debug_file,
    print_active_orders,
    print_market_context,
    maintain_unfilled_live_orders,
)
from custom.btc_agent.paper_state import ActivePaperOrder


class TestBtcMain(unittest.TestCase):
    def test_run_once_catches_transient_request_exception(self):
        with patch(
            "custom.btc_agent.main._run_once_core",
            side_effect=requests.ReadTimeout("CLOB timeout"),
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            run_once()

        output = stdout.getvalue()
        self.assertIn("WARNING: Network connection issue encountered during core tick: CLOB timeout", output)
        self.assertIn("Retrying on the next tick interval", output)

    def test_run_once_catches_unexpected_tick_exception(self):
        with patch(
            "custom.btc_agent.main._run_once_core",
            side_effect=ValueError("bad tick"),
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            run_once()

        output = stdout.getvalue()
        self.assertIn("ERROR: Unexpected internal engine error during core tick: bad tick", output)
        self.assertIn("Skipping active tick frame", output)

    def test_print_active_orders_shows_unfilled_for_pending_live_order(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1770000000",
            market_title="Bitcoin Up or Down",
            side="UP",
            shares=3.0,
            entry_price=0.67,
            token_id="up-token",
            target_btc_price=80000.0,
            entry_btc_price=80010.0,
            actual_fill_price=None,
            api_order_state="ORDER_STATE_OPEN",
            api_state="ORDER_STATE_OPEN",
        )

        observed_at = datetime.fromtimestamp(1_770_000_010, tz=timezone.utc)
        with patch("custom.btc_agent.main.get_active_orders", return_value=[order]), patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout:
            print_active_orders(80020.0, observed_at=observed_at)

        output = stdout.getvalue()
        self.assertIn("observed_at            = 2026-02-02T02:40:10+00:00", output)
        self.assertIn("order_1_polymarket_api_state = ORDER_STATE_OPEN", output)
        self.assertIn("order_1_internal_filled_flag = False", output)
        self.assertIn(
            "order_1_position_state = UNFILLED (Polymarket State: ORDER_STATE_OPEN)",
            output,
        )

    def test_print_active_orders_uses_filled_flag_even_with_fill_price(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1770000000",
            market_title="Bitcoin Up or Down",
            side="UP",
            shares=3.0,
            entry_price=0.67,
            token_id="up-token",
            target_btc_price=80000.0,
            entry_btc_price=80010.0,
            actual_fill_price=0.71,
            filled=False,
            api_order_state="ORDER_STATE_PARTIALLY_FILLED",
            api_state="ORDER_STATE_PARTIALLY_FILLED",
        )

        with patch("custom.btc_agent.main.get_active_orders", return_value=[order]), patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout:
            print_active_orders(80020.0)

        self.assertIn(
            "order_1_position_state = UNFILLED (Polymarket State: ORDER_STATE_PARTIALLY_FILLED)",
            stdout.getvalue(),
        )

    def test_print_market_context_shows_observed_at_without_debug(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1770000000",
            settlement_threshold=80000.0,
            up_market_probability=0.61,
            down_market_probability=0.39,
        )
        observed_at = datetime.fromtimestamp(1_770_000_010, tz=timezone.utc)

        with patch(
            "custom.btc_agent.main.get_realized_pnl_snapshot",
            return_value=(0.0, 0.0),
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            print_market_context(market, debug=False, observed_at=observed_at)

        output = stdout.getvalue()
        self.assertIn("Market:", output)
        self.assertIn("observed_at            = 2026-02-02T02:40:10+00:00", output)
        self.assertIn("slug                  = btc-updown-5m-1770000000", output)

    def test_maintain_unfilled_live_orders_returns_naturally_filled_orders(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1779999900",
            market_title="Bitcoin Up or Down",
            side="UP",
            shares=5.0,
            entry_price=0.67,
            token_id="up-token",
            target_btc_price=80000.0,
            entry_btc_price=80010.0,
            live_order_id="order-filled",
            placed_at=datetime.now(timezone.utc) - timedelta(seconds=30),
            filled=True,
            actual_fill_price=0.67,
        )

        with patch(
            "custom.btc_agent.main.refresh_live_order_fill_status",
            return_value=True,
        ), patch("custom.btc_agent.executor.cancel_live_order") as cancel_order:
            result = maintain_unfilled_live_orders([order], SimpleNamespace())

        self.assertEqual(result, [order])
        cancel_order.assert_not_called()

    def test_maintain_unfilled_live_orders_cancels_after_timeout_and_logs(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1779999901",
            market_title="Bitcoin Up or Down",
            side="DOWN",
            shares=5.0,
            entry_price=0.67,
            token_id="down-token",
            target_btc_price=80000.0,
            entry_btc_price=79990.0,
            live_order_id="order-timeout",
            placed_at=datetime.now(timezone.utc) - timedelta(seconds=30),
            filled=False,
            actual_fill_price=None,
        )
        log_path = "/tmp/completed_orders/completed_order_unfilled_down_1779999901.txt"

        with patch.dict(os.environ, {"CANCEL_UNFILLED_TIMER": "10"}), patch(
            "custom.btc_agent.main.refresh_live_order_fill_status",
            return_value=False,
        ), patch(
            "custom.btc_agent.executor.cancel_live_order",
            return_value=True,
        ) as cancel_order, patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/tmp",
        ), patch(
            "custom.btc_agent.main.append_period_analysis_sample"
        ) as append_analysis, patch(
            "custom.btc_agent.main.revert_executed_trade"
        ) as revert_trade, patch(
            "custom.btc_agent.main.os.replace",
            wraps=os.replace,
        ), patch(
            "custom.btc_agent.main.time.sleep"
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = maintain_unfilled_live_orders([order], SimpleNamespace())

        self.assertEqual(result, [])
        cancel_order.assert_called_once_with("order-timeout")
        self.assertFalse(order.filled)
        self.assertEqual(order.api_order_state, "CANCELED")
        self.assertEqual(order.api_state, "CANCELED")
        self.assertIn("CANCELED: Unfilled order order-timeout", stdout.getvalue())
        append_analysis.assert_called_once()
        self.assertEqual(append_analysis.call_args.kwargs["phase"], "LIVE_ORDER_CANCELED_TIMEOUT")
        self.assertEqual(append_analysis.call_args.kwargs["outcome_label"], "unfilled")
        revert_trade.assert_called_once_with(order)

        with open(log_path, encoding="utf-8") as order_log:
            content = order_log.read()

        self.assertIn("phase=LIVE_ORDER_CANCELED_TIMEOUT", content)
        self.assertIn(
            "reason=Unfilled order order-timeout was placed but cancelled because it was not filled within 10.0 seconds.",
            content,
        )

    def test_maintain_unfilled_live_orders_preserves_post_cancel_ghost_fill(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1779999905",
            market_title="Bitcoin Up or Down",
            side="UP",
            shares=5.0,
            entry_price=0.67,
            token_id="up-token",
            target_btc_price=80000.0,
            entry_btc_price=80010.0,
            live_order_id="order-ghost-fill",
            placed_at=datetime.now(timezone.utc) - timedelta(seconds=30),
            filled=False,
            actual_fill_price=None,
        )

        with patch.dict(os.environ, {"CANCEL_UNFILLED_TIMER": "10"}), patch(
            "custom.btc_agent.main.refresh_live_order_fill_status",
            side_effect=[False, False, True],
        ) as refresh_status, patch(
            "custom.btc_agent.executor.cancel_live_order",
            return_value=True,
        ) as cancel_order, patch(
            "custom.btc_agent.main.revert_executed_trade"
        ) as revert_trade, patch(
            "custom.btc_agent.main.time.sleep"
        ) as sleep:
            result = maintain_unfilled_live_orders([order], SimpleNamespace())

        self.assertEqual(result, [order])
        self.assertEqual(refresh_status.call_count, 3)
        cancel_order.assert_called_once_with("order-ghost-fill")
        sleep.assert_called_once_with(1.5)
        revert_trade.assert_not_called()

    def test_maintain_unfilled_live_orders_releases_slot_only_after_confirmed_cancel(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1779999906",
            market_title="Bitcoin Up or Down",
            side="DOWN",
            shares=5.0,
            entry_price=0.67,
            token_id="down-token",
            target_btc_price=80000.0,
            entry_btc_price=79990.0,
            live_order_id="order-confirmed-cancel",
            placed_at=datetime.now(timezone.utc) - timedelta(seconds=30),
            filled=False,
            actual_fill_price=None,
        )
        call_count = {"refresh": 0}

        def refresh_status(_order):
            call_count["refresh"] += 1
            if call_count["refresh"] == 3:
                _order.api_order_state = "ORDER_STATE_CANCELED"
            return False

        with patch.dict(os.environ, {"CANCEL_UNFILLED_TIMER": "10"}), patch(
            "custom.btc_agent.main.refresh_live_order_fill_status",
            side_effect=refresh_status,
        ) as refresh_status_mock, patch(
            "custom.btc_agent.executor.cancel_live_order",
            return_value=True,
        ) as cancel_order, patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/tmp",
        ), patch(
            "custom.btc_agent.main.append_period_analysis_sample"
        ), patch(
            "custom.btc_agent.main.revert_executed_trade"
        ) as revert_trade, patch(
            "custom.btc_agent.main.os.replace",
            wraps=os.replace,
        ), patch(
            "custom.btc_agent.main.time.sleep"
        ) as sleep:
            result = maintain_unfilled_live_orders([order], SimpleNamespace())

        self.assertEqual(result, [])
        self.assertEqual(refresh_status_mock.call_count, 3)
        cancel_order.assert_called_once_with("order-confirmed-cancel")
        sleep.assert_called_once_with(1.5)
        self.assertEqual(order.api_order_state, "CANCELED")
        self.assertEqual(order.api_state, "CANCELED")
        revert_trade.assert_called_once_with(order)

    def test_maintain_unfilled_live_orders_final_status_check_blocks_cancel(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1779999903",
            market_title="Bitcoin Up or Down",
            side="UP",
            shares=5.0,
            entry_price=0.67,
            token_id="up-token",
            target_btc_price=80000.0,
            entry_btc_price=80010.0,
            live_order_id="order-race-filled",
            placed_at=datetime.now(timezone.utc) - timedelta(seconds=30),
            filled=False,
            actual_fill_price=None,
        )

        with patch.dict(os.environ, {"CANCEL_UNFILLED_TIMER": "10"}), patch(
            "custom.btc_agent.main.refresh_live_order_fill_status",
            side_effect=[False, True],
        ) as refresh_status, patch("custom.btc_agent.executor.cancel_live_order") as cancel_order:
            result = maintain_unfilled_live_orders([order], SimpleNamespace())

        self.assertEqual(result, [order])
        self.assertEqual(refresh_status.call_count, 2)
        cancel_order.assert_not_called()

    def test_maintain_unfilled_live_orders_skips_matched_state(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1779999904",
            market_title="Bitcoin Up or Down",
            side="UP",
            shares=5.0,
            entry_price=0.67,
            token_id="up-token",
            target_btc_price=80000.0,
            entry_btc_price=80010.0,
            live_order_id="order-matched",
            placed_at=datetime.now(timezone.utc) - timedelta(seconds=30),
            filled=False,
            actual_fill_price=None,
            api_order_state="MATCHED",
            api_state="MATCHED",
        )

        with patch.dict(os.environ, {"CANCEL_UNFILLED_TIMER": "10"}), patch(
            "custom.btc_agent.main.refresh_live_order_fill_status",
            return_value=False,
        ), patch("custom.btc_agent.executor.cancel_live_order") as cancel_order:
            result = maintain_unfilled_live_orders([order], SimpleNamespace())

        self.assertEqual(result, [])
        cancel_order.assert_not_called()

    def test_maintain_unfilled_live_orders_skips_already_canceled_order(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1779999902",
            market_title="Bitcoin Up or Down",
            side="UP",
            shares=5.0,
            entry_price=0.67,
            token_id="up-token",
            target_btc_price=80000.0,
            entry_btc_price=80010.0,
            live_order_id="order-canceled",
            placed_at=datetime.now(timezone.utc) - timedelta(seconds=30),
            filled=False,
            actual_fill_price=None,
            api_order_state="ORDER_STATE_CANCELED",
            api_state="ORDER_STATE_CANCELED",
        )

        with patch.dict(os.environ, {"CANCEL_UNFILLED_TIMER": "10"}), patch(
            "custom.btc_agent.main.refresh_live_order_fill_status",
            return_value=False,
        ), patch("custom.btc_agent.executor.cancel_live_order") as cancel_order:
            result = maintain_unfilled_live_orders([order], SimpleNamespace())

        self.assertEqual(result, [])
        cancel_order.assert_not_called()

    def test_append_period_analysis_sample_throttles_and_forces_trade_records(self):
        from custom.btc_agent import main as main_module

        main_module._PERIOD_ANALYSIS_LAST_SAMPLE_BY_SLUG.clear()
        analysis_glob = "/tmp/completed_orders/period_analysis_*.json"
        for path in glob.glob(analysis_glob):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

        market = SimpleNamespace(
            slug="btc-updown-5m-1777513500",
            title="Bitcoin Up or Down",
            settlement_threshold=80000.0,
            up_market_probability=0.62,
            down_market_probability=0.38,
            up_token_id="up-token",
            down_token_id="down-token",
        )
        features = SimpleNamespace(
            price_usd=80018.0,
            delta_from_previous_tick=2.0,
            momentum_1m=6.0,
            momentum_5m=12.0,
            velocity_15s=2.0,
            velocity_30s=4.0,
            momentum_acceleration=1.0,
            volatility_5m=14.0,
            rsi_9=64.0,
            rsi_14=58.0,
            rsi_speed_divergence=6.0,
            ema_9=80010.0,
            ema_21=80000.0,
            ema_alignment=True,
            ema_cross_direction="bullish",
            adx_14=31.0,
            atr_14=8.0,
            window_open_price=79990.0,
            delta_pct_from_window_open=0.00035,
            trailing_5m_open_price=79980.0,
            delta_pct_from_trailing_5m_open=0.00048,
            consecutive_flat_ticks=0,
            consecutive_directional_ticks=4,
            last_10_ticks_direction="UUUD",
            retained_sample_count=25,
            window_sample_count=12,
            trailing_5m_sample_count=20,
        )
        decision = SimpleNamespace(
            side="UP",
            confidence=0.72,
            max_price_to_pay=1.0,
            reason="test",
            raw_response_text='{"decision":"UP"}',
        )
        result = SimpleNamespace(
            executed=False,
            submission_accepted=False,
            side="UP",
            size=0.0,
            price=0.64,
            token_id="up-token",
            reason="test rejection",
            veto_flags=["test_veto"],
            live_order_id=None,
            quoted_price_at_entry=None,
            actual_fill_price=None,
            realized_slippage_bps=None,
            order_latency_ms=None,
            book_depth_at_fill=None,
            shares_requested=7.0,
        )

        with patch("custom.btc_agent.main.os.getcwd", return_value="/tmp"), patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(
                period_analysis_interval_seconds=30,
                paper_trading=True,
                shares_per_trade=7.0,
                min_confidence=0.64,
            ),
        ), patch(
            "custom.btc_agent.main.get_llm_config",
            return_value=SimpleNamespace(engine="openai", model="gpt-4.1-mini"),
        ), patch(
            "custom.btc_agent.main.get_realized_pnl_snapshot",
            return_value=(1.25, 0.50),
        ), patch(
            "custom.btc_agent.main._SESSION_WIN_TRADES",
            2,
        ), patch(
            "custom.btc_agent.main._SESSION_LOSS_TRADES",
            1,
        ):
            append_period_analysis_sample(
                market,
                phase="PRE_ORDER_TICK",
                features=features,
                decision=decision,
                observed_at=datetime.fromtimestamp(1777513510, tz=timezone.utc),
            )
            append_period_analysis_sample(
                market,
                phase="PRE_ORDER_TICK",
                features=features,
                decision=decision,
                observed_at=datetime.fromtimestamp(1777513515, tz=timezone.utc),
            )
            append_period_analysis_sample(
                market,
                phase="TRADE_EVALUATION",
                features=features,
                decision=decision,
                result=result,
                observed_at=datetime.fromtimestamp(1777513516, tz=timezone.utc),
                force=True,
            )

        analysis_paths = glob.glob(analysis_glob)
        self.assertEqual(analysis_paths, ["/tmp/completed_orders/period_analysis_1777513500.json"])
        with open(analysis_paths[0], encoding="utf-8") as analysis_file:
            records = [json.loads(line) for line in analysis_file if line.strip()]

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["llm"]["model"], "gpt-4.1-mini")
        self.assertEqual(records[0]["market"]["slug"], market.slug)
        self.assertEqual(records[0]["features"]["btc_price"], 80018.0)
        self.assertEqual(records[0]["pnl"]["session_win_trades"], 2)
        self.assertEqual(records[0]["pnl"]["session_loss_trades"], 1)
        self.assertEqual(records[0]["pnl"]["session_resolved_trades"], 3)
        self.assertEqual(records[1]["phase"], "TRADE_EVALUATION")
        self.assertEqual(records[1]["execution"]["veto_flags"], ["test_veto"])

        for path in analysis_paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    TEST_TIMESTAMPS = (
        "1777056000",
        "1777513500",
        "1777513800",
        "1777513811",
        "1777513999",
        "1777675200",
        "1778208300",
        "1779999900",
        "1779999901",
        "1779999902",
        "1779999906",
        "1999999997",
        "1999999998",
        "1999999999",
    )

    def setUp(self):
        self._live_probability_patcher = patch(
            "custom.btc_agent.market_lookup.fetch_live_market_probabilities_from_clob_ws",
            return_value=(0.50, 0.50),
        )
        self._live_probability_patcher.start()

    def tearDown(self):
        self._live_probability_patcher.stop()
        from custom.btc_agent import main as main_module
        main_module._SESSION_PENDING_EXIT_AFTER_PERIOD = False
        self._cleanup_test_artifacts()

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_test_artifacts()

    @classmethod
    def _cleanup_test_artifacts(cls):
        paths = [
            "/appl/agents/logs/priceToBeatDebug.txt",
            "/appl/agents/logs/priceToBeatDebugPg2.txt",
            "/appl/agents/logs/priceToBeatDebugPg3.txt",
            "/appl/agents/completed_orders/period_analysis.json",
            "/tmp/completed_orders/period_analysis.json",
        ]
        paths.extend(glob.glob("/tmp/completed_orders/period_analysis_*.json"))
        for timestamp in cls.TEST_TIMESTAMPS:
            paths.extend(glob.glob(f"/appl/agents/completed_orders/*{timestamp}*.txt"))
            paths.extend(glob.glob(f"/appl/agents/completed_orders/period_analysis_{timestamp}.json"))
            paths.extend(glob.glob(f"/tmp/completed_orders/*{timestamp}*.txt"))

        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def test_enforce_session_loss_trade_limit_exits_when_cap_reached(self):
        with patch(
            "custom.btc_agent.main._SESSION_LOSS_TRADES",
            3,
        ), patch(
            "custom.btc_agent.main.sys.exit",
            side_effect=SystemExit(0),
        ) as mock_exit:
            with self.assertRaises(SystemExit):
                enforce_session_loss_trade_limit(
                    SimpleNamespace(max_net_losses_per_run=3)
                )

        mock_exit.assert_called_once_with(0)

    def test_enforce_session_loss_trade_limit_does_not_exit_below_cap(self):
        with patch(
            "custom.btc_agent.main._SESSION_LOSS_TRADES",
            2,
        ), patch(
            "custom.btc_agent.main.sys.exit",
        ) as mock_exit:
            enforce_session_loss_trade_limit(
                SimpleNamespace(max_net_losses_per_run=3)
            )

        mock_exit.assert_not_called()

    def test_finalize_completed_orders_win_decrements_net_loss_count(self):
        from custom.btc_agent import main as main_module

        main_module._SESSION_WIN_TRADES = 0
        main_module._SESSION_LOSS_TRADES = 1
        try:
            with patch(
                "custom.btc_agent.main.append_completed_order_tick",
                return_value="win",
            ), patch("custom.btc_agent.main.record_realized_pnl_for_order"):
                completed_losses = main_module.finalize_completed_orders([object()], 77770.0)

            self.assertEqual(completed_losses, 0)
            self.assertEqual(main_module._SESSION_WIN_TRADES, 1)
            self.assertEqual(main_module._SESSION_LOSS_TRADES, 0)
        finally:
            main_module._SESSION_WIN_TRADES = 0
            main_module._SESSION_LOSS_TRADES = 0

    def test_finalize_completed_orders_win_does_not_decrement_net_loss_below_zero(self):
        from custom.btc_agent import main as main_module

        main_module._SESSION_WIN_TRADES = 0
        main_module._SESSION_LOSS_TRADES = 0
        try:
            with patch(
                "custom.btc_agent.main.append_completed_order_tick",
                return_value="win",
            ), patch("custom.btc_agent.main.record_realized_pnl_for_order"):
                completed_losses = main_module.finalize_completed_orders([object()], 77770.0)

            self.assertEqual(completed_losses, 0)
            self.assertEqual(main_module._SESSION_WIN_TRADES, 1)
            self.assertEqual(main_module._SESSION_LOSS_TRADES, 0)
        finally:
            main_module._SESSION_WIN_TRADES = 0
            main_module._SESSION_LOSS_TRADES = 0

    def test_should_not_log_failed_order_attempt_for_paper_trade_rejection(self):
        cfg = SimpleNamespace(paper_trading=True)
        decision = SimpleNamespace(side="UP")
        result = SimpleNamespace(
            executed=False,
            token_id="up-token",
            reason="Quote-floor veto blocked low-probability reversal trade",
        )

        self.assertTrue(_should_log_failed_order_attempt(cfg, decision, result))

    def test_should_log_failed_order_attempt_for_live_submission_failure(self):
        cfg = SimpleNamespace(paper_trading=False)
        decision = SimpleNamespace(side="UP")
        result = SimpleNamespace(
            executed=False,
            token_id="up-token",
            reason="FOK order could not be fully filled in the final deadline window",
        )

        self.assertTrue(_should_log_failed_order_attempt(cfg, decision, result))

    def test_append_failed_order_attempt_marks_paper_rejection_without_submission(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1999999998",
            title="Bitcoin Up or Down",
            settlement_threshold=77763.01,
        )
        decision = SimpleNamespace(
            side="UP",
            confidence=0.8,
            max_price_to_pay=1.0,
            prompt_text="SYSTEM PROMPT:\nfoo\n\nUSER PROMPT:\nbar",
            raw_response_text='{"decision":"UP","confidence":0.8,"max_price_to_pay":1.0,"reason":"test"}',
        )
        result = SimpleNamespace(
            executed=False,
            side="UP",
            size=0.0,
            price=0.62,
            token_id="up-token",
            reason="Quote-floor veto blocked low-probability reversal trade",
            quoted_price_at_entry=0.61,
            actual_fill_price=None,
            realized_slippage_bps=None,
            order_latency_ms=0,
            book_depth_at_fill=4.5,
            shares_requested=3.0,
        )

        with patch("custom.btc_agent.main.os.getcwd", return_value="/appl/agents"):
            append_failed_order_attempt(
                market,
                decision,
                result,
                paper_trading=True,
                observed_at=datetime.now(timezone.utc),
            )

        with open(
            "/appl/agents/completed_orders/completed_order_attempt_1999999998.txt",
            encoding="utf-8",
        ) as attempt_file:
            content = attempt_file.read()

        self.assertIn("phase=PAPER_REJECTED_BEFORE_EXECUTION", content)
        self.assertIn("attempt_class=paper_validation_rejection", content)
        self.assertIn("paper_trading=True", content)
        self.assertIn("order_submission_attempted=false", content)
        self.assertIn("min_confidence=", content)
        self.assertIn("veto_threshold_delta=", content)
        self.assertIn("llm_raw_reasoning=", content)
        self.assertIn("market_impact_ratio=", content)
        self.assertIn("veto_flags=", content)
        self.assertIn("llm_prompt_start", content)
        self.assertIn("SYSTEM PROMPT:", content)
        self.assertIn("llm_prompt_end", content)
        self.assertIn("llm_raw_response_start", content)
        self.assertIn('"decision":"UP"', content)
        self.assertIn("llm_raw_response_end", content)
        self.assertIn("quoted_price_at_entry=0.610", content)
        self.assertIn("book_depth_at_fill=4.500", content)
        self.assertIn("shares_requested=3.000", content)

    def test_append_failed_live_order_copies_period_context_and_failure_reason(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1999999998",
            title="Bitcoin Up or Down",
            settlement_threshold=77763.01,
            up_market_probability=0.68,
            down_market_probability=0.32,
        )
        decision = SimpleNamespace(
            side="UP",
            confidence=0.72,
            max_price_to_pay=1.0,
            reason="trend-following trade",
            prompt_text="SYSTEM PROMPT:\nfoo\n\nUSER PROMPT:\nbar",
            raw_response_text='{"decision":"UP","confidence":0.72,"max_price_to_pay":1.0,"reason":"trend-following trade"}',
        )
        features = SimpleNamespace(
            as_of=datetime.now(timezone.utc),
            price_usd=77780.0,
            delta_from_previous_tick=1.0,
            momentum_1m=2.0,
            momentum_5m=3.0,
            velocity_15s=1.0,
            velocity_30s=1.5,
            momentum_acceleration=-0.5,
            volatility_5m=5.0,
            consecutive_flat_ticks=0,
            consecutive_directional_ticks=3,
            last_10_ticks_direction="UUUD",
            rsi_9=65.0,
            rsi_speed_divergence=1.5,
            ema_9=77770.0,
            ema_21=77760.0,
            ema_alignment=True,
            ema_cross_direction="bullish",
            adx_14=38.5,
            atr_14=6.0,
        )
        up_snapshot = SimpleNamespace(
            buy_quote=0.74,
            reference_price=0.74,
            target_limit_price=0.74,
            recommended_limit_price=0.74,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.73,
            best_ask=0.74,
            best_bid_size=100.0,
            best_ask_size=100.0,
            spread=0.01,
            spread_bps=135.0,
            top_level_book_imbalance=0.51,
            imbalance_pressure=0.02,
        )
        down_snapshot = SimpleNamespace(
            buy_quote=0.26,
            reference_price=0.26,
            target_limit_price=0.26,
            recommended_limit_price=0.26,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.25,
            best_ask=0.26,
            best_bid_size=100.0,
            best_ask_size=100.0,
            spread=0.01,
            spread_bps=400.0,
            top_level_book_imbalance=0.49,
            imbalance_pressure=-0.02,
        )

        with patch("custom.btc_agent.main.os.getcwd", return_value="/appl/agents"):
            append_pending_period_tick_analysis(
                market,
                up_snapshot=up_snapshot,
                down_snapshot=down_snapshot,
                features=features,
                decision=decision,
                observed_at=features.as_of,
            )
            append_failed_live_order(
                market,
                decision,
                "Live order submission failed: PolyApiException[status_code=None, error_message=Request exception!]",
                features=features,
                up_snapshot=up_snapshot,
                down_snapshot=down_snapshot,
                observed_at=features.as_of,
            )

        with open(
            "/appl/agents/completed_orders/failed_order_1999999998.txt",
            encoding="utf-8",
        ) as failure_file:
            content = failure_file.read()

        self.assertIn("phase=PRE_ORDER_TICK", content)
        self.assertIn("phase=LIVE_ORDER_RUNTIME_FAILURE", content)
        self.assertIn("failure_reason=Live order submission failed: PolyApiException[status_code=None, error_message=Request exception!]", content)
        self.assertIn("llm_raw_response_start", content)

    def test_run_once_logs_live_submission_failure_without_exiting(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1999999998",
            title="Bitcoin Up or Down",
            settlement_threshold=77763.01,
            up_token_id="up-token",
            down_token_id="down-token",
            start_ts=1999999800,
            up_market_probability=0.74,
            down_market_probability=0.26,
        )
        state = SimpleNamespace(trades_executed=0, active_orders=[])
        features = SimpleNamespace(
            as_of=datetime.now(timezone.utc),
            price_usd=77780.0,
            delta_from_previous_tick=1.0,
            momentum_1m=2.0,
            momentum_5m=3.0,
            velocity_15s=1.0,
            velocity_30s=1.5,
            momentum_acceleration=-0.5,
            volatility_5m=5.0,
            consecutive_flat_ticks=0,
            consecutive_directional_ticks=3,
            last_10_ticks_direction="UUUD",
            rsi_9=65.0,
            rsi_14=60.0,
            rsi_speed_divergence=1.5,
            ema_9=77770.0,
            ema_21=77760.0,
            ema_alignment=True,
            ema_cross_direction="bullish",
            adx_14=38.5,
            atr_14=6.0,
            window_open_price=77770.0,
            delta_pct_from_window_open=0.001,
            trailing_5m_open_price=77760.0,
            delta_pct_from_trailing_5m_open=0.002,
        )
        decision = SimpleNamespace(
            side="UP",
            confidence=0.72,
            max_price_to_pay=1.0,
            reason="trend-following trade",
            prompt_text=None,
            raw_response_text='{"decision":"UP","confidence":0.72,"max_price_to_pay":1.0,"reason":"trend-following trade"}',
        )
        up_snapshot = SimpleNamespace(
            token_id="up-token",
            ok_to_submit=True,
            buy_quote=0.74,
            midpoint=0.74,
            last_trade_price=0.74,
            reference_price=0.74,
            target_limit_price=0.74,
            recommended_limit_price=0.74,
            submit_reason="ok",
            best_bid=0.73,
            best_ask=0.74,
            tick_size=0.01,
            spread=0.01,
            best_bid_size=100.0,
            best_ask_size=100.0,
            spread_bps=135.0,
            top_level_book_imbalance=0.51,
            imbalance_pressure=0.02,
        )
        down_snapshot = SimpleNamespace(
            token_id="down-token",
            ok_to_submit=True,
            buy_quote=0.26,
            midpoint=0.26,
            last_trade_price=0.26,
            reference_price=0.26,
            target_limit_price=0.26,
            recommended_limit_price=0.26,
            submit_reason="ok",
            best_bid=0.25,
            best_ask=0.26,
            tick_size=0.01,
            spread=0.01,
            best_bid_size=100.0,
            best_ask_size=100.0,
            spread_bps=400.0,
            top_level_book_imbalance=0.49,
            imbalance_pressure=-0.02,
        )

        with ExitStack() as stack:
            stack.enter_context(patch("custom.btc_agent.main._FIRST_LOOP", False))
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_trading_config",
                    return_value=SimpleNamespace(
                        debug=False,
                        debug_price_to_beat=False,
                        max_trades_per_period=1,
                        max_losses_per_run=0,
                        minimum_wallet_balance=0.0,
                        paper_trading=False,
                        use_recommended_limit=True,
                    ),
                )
            )
            stack.enter_context(patch("custom.btc_agent.main.find_current_btc_updown_market", return_value=market))
            stack.enter_context(patch("custom.btc_agent.main.sync_period_state", return_value=False))
            stack.enter_context(patch("custom.btc_agent.main.get_state", return_value=state))
            stack.enter_context(patch("custom.btc_agent.main.resolve_price_to_beat_with_retries", return_value=market))
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_token_quote_snapshot",
                    side_effect=[up_snapshot, down_snapshot],
                )
            )
            stack.enter_context(patch("custom.btc_agent.main.build_btc_features", return_value=features))
            stack.enter_context(patch("custom.btc_agent.main.get_feature_readiness", return_value=(True, None)))
            stack.enter_context(patch("custom.btc_agent.main.decide_trade", return_value=decision))
            stack.enter_context(patch("custom.btc_agent.main.get_decision_quote_snapshot", return_value=up_snapshot))
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.maybe_execute_trade",
                    side_effect=RuntimeError(
                        "Live order submission failed: PolyApiException[status_code=None, error_message=Request exception!]"
                    ),
                )
            )
            stack.enter_context(patch("custom.btc_agent.main.print_account_snapshot_from_snapshot"))
            stack.enter_context(patch("custom.btc_agent.main.enforce_minimum_wallet_balance"))
            stack.enter_context(patch("custom.btc_agent.main.print_market_context"))
            stack.enter_context(patch("custom.btc_agent.main.print_quote_snapshot_from_snapshot"))
            stack.enter_context(patch("custom.btc_agent.main.print_features"))
            stack.enter_context(patch("custom.btc_agent.main.print_llm_decision"))
            stack.enter_context(patch("custom.btc_agent.main.print_trade_execution_result"))
            mock_exit = stack.enter_context(patch("custom.btc_agent.main.sys.exit"))
            stack.enter_context(patch("custom.btc_agent.main.os.getcwd", return_value="/appl/agents"))
            run_once()

        mock_exit.assert_not_called()
        self.assertTrue(os.path.exists("/appl/agents/completed_orders/failed_order_1999999998.txt"))

    def test_wait_for_next_tick_or_quit_returns_true_when_q_requested(self):
        quit_monitor = SimpleNamespace(poll_quit_requested=lambda: True)

        should_quit = wait_for_next_tick_or_quit(
            30,
            quit_monitor=quit_monitor,
            poll_interval_seconds=0.01,
        )

        self.assertTrue(should_quit)

    def test_has_valid_price_to_beat_rejects_none_and_small_values(self):
        self.assertFalse(has_valid_price_to_beat(None))
        self.assertFalse(has_valid_price_to_beat(1))
        self.assertFalse(has_valid_price_to_beat(5))

    def test_has_valid_price_to_beat_accepts_realistic_btc_values(self):
        self.assertTrue(has_valid_price_to_beat(78218.01972274295))

    def test_print_features_outputs_primary_btc_price(self):
        features = SimpleNamespace(
            price_usd=80382.04,
            delta_from_previous_tick=5.0,
            momentum_1m=7.0,
            momentum_5m=10.0,
            velocity_15s=4.0,
            velocity_30s=6.0,
            momentum_acceleration=-2.0,
            volatility_5m=22.0,
            consecutive_flat_ticks=0,
            consecutive_directional_ticks=3,
            rsi_9=61.0,
            rsi_14=55.0,
            rsi_speed_divergence=6.0,
            ema_9=74980.0,
            ema_21=74960.0,
            ema_alignment=True,
            ema_cross_direction="bullish",
            adx_14=31.0,
            atr_14=12.0,
            last_10_ticks_direction="UUUDUUUUUU",
        )

        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            print_features(features, debug=False)

        content = stdout.getvalue()
        self.assertIn("btc_price             = 80382.04", content)
        self.assertNotIn("btc_price_poly", content)

    def test_print_llm_decision_outputs_prompt_in_debug_mode(self):
        decision = SimpleNamespace(
            side="UP",
            confidence=0.8,
            max_price_to_pay=1.0,
            reason="test",
            prompt_text="SYSTEM PROMPT:\nfoo\n\nUSER PROMPT:\nbar",
            raw_response_text='{"decision":"UP","confidence":0.8,"max_price_to_pay":1.0,"reason":"test"}',
        )
        market = SimpleNamespace(
            settlement_threshold=100.0,
            up_market_probability=0.65,
            down_market_probability=0.35,
        )
        features = SimpleNamespace(
            price_usd=100.5,
            adx_14=40.0,
        )

        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            print_llm_decision(decision, market=market, features=features, debug=True)

        content = stdout.getvalue()
        self.assertIn("LLM prompt:", content)
        self.assertIn("SYSTEM PROMPT:", content)
        self.assertIn("USER PROMPT:", content)
        self.assertIn("LLM raw response:", content)
        self.assertIn('"decision":"UP"', content)

    def test_append_completed_order_tick_writes_completed_order_file(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1777513800",
            market_title="Bitcoin Up or Down",
            side="UP",
            shares=5.0,
            entry_price=0.45,
            token_id="up-token",
            target_btc_price=77763.01,
            entry_btc_price=77760.00,
            quoted_price_at_entry=0.44,
            actual_fill_price=0.45,
            filled=True,
            realized_slippage_bps=227.27,
            order_latency_ms=342,
            book_depth_at_fill=12.5,
            shares_requested=5.0,
            api_order_state="ORDER_STATE_FILLED",
            api_state="ORDER_STATE_FILLED",
            llm_prompt_text="SYSTEM PROMPT:\nfoo\n\nUSER PROMPT:\nbar",
            llm_raw_response_text='{"decision":"UP","confidence":0.8,"max_price_to_pay":1.0,"reason":"test"}',
        )

        with patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/appl/agents",
        ):
            append_completed_order_tick(order, current_btc_price=77770.0, phase="PLACED")

        with open(
            "/appl/agents/completed_orders/completed_order_1777513800.txt",
            encoding="utf-8",
        ) as order_file:
            content = order_file.read()

        self.assertIn("market_slug=btc-updown-5m-1777513800", content)
        self.assertIn("phase=PLACED", content)
        self.assertIn("position_state=WINNING", content)
        self.assertIn("btc_move_from_entry=10.00", content)
        self.assertIn("btc_gap_to_target=6.99", content)
        self.assertIn("market_time_remaining_mmss=", content)
        self.assertIn("polymarket_api_state=ORDER_STATE_FILLED", content)
        self.assertIn("internal_filled_flag=True", content)
        self.assertIn("order_1_polymarket_api_state=ORDER_STATE_FILLED", content)
        self.assertIn("order_1_internal_filled_flag=True", content)
        self.assertIn("outcome_label=win", content)
        self.assertIn("quoted_price_at_entry=0.440", content)
        self.assertIn("actual_fill_price=0.450", content)
        self.assertIn("realized_slippage_bps=227.270", content)
        self.assertIn("order_latency_ms=342", content)
        self.assertIn("book_depth_at_fill=12.500", content)
        self.assertIn("shares_requested=5.000", content)
        self.assertIn("llm_prompt_start", content)
        self.assertIn("SYSTEM PROMPT:", content)
        self.assertIn("llm_prompt_end", content)
        self.assertIn("llm_raw_response_start", content)
        self.assertIn('"decision":"UP"', content)
        self.assertIn("llm_raw_response_end", content)

    def test_append_pending_period_tick_analysis_writes_llm_prompt_when_present(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1777513999",
            title="Bitcoin Up or Down",
            settlement_threshold=77763.01,
            up_market_probability=0.51,
            down_market_probability=0.49,
        )
        decision = SimpleNamespace(
            side="UP",
            confidence=0.8,
            max_price_to_pay=1.0,
            reason="test",
            prompt_text="SYSTEM PROMPT:\nfoo\n\nUSER PROMPT:\nbar",
            raw_response_text='{"decision":"UP","confidence":0.8,"max_price_to_pay":1.0,"reason":"test"}',
        )

        with patch("custom.btc_agent.main.os.getcwd", return_value="/appl/agents"):
            append_pending_period_tick_analysis(
                market,
                decision=decision,
                observed_at=datetime.now(timezone.utc),
            )

        with open(
            "/appl/agents/completed_orders/pending_period_1777513999.txt",
            encoding="utf-8",
        ) as pending_file:
            content = pending_file.read()

        self.assertIn("llm_prompt_start", content)
        self.assertIn("SYSTEM PROMPT:", content)
        self.assertIn("USER PROMPT:", content)
        self.assertIn("llm_prompt_end", content)
        self.assertIn("llm_raw_response_start", content)
        self.assertIn('"decision":"UP"', content)
        self.assertIn("llm_raw_response_end", content)

    def test_regime_fingerprint_uses_price_to_beat_to_avoid_false_weak_down_label(self):
        features = SimpleNamespace(
            price_usd=78134.0,
            delta_pct_from_window_open=-0.000155,
            momentum_5m=-2.0,
            volatility_5m=9.0,
            rsi_9=82.0,
            rsi_14=76.0,
            atr_14=20.0,
            adx_14=38.0,
            rsi_speed_divergence=6.0,
            velocity_15s=4.0,
            velocity_30s=7.0,
            consecutive_flat_ticks=0,
            consecutive_directional_ticks=4,
        )
        observed_at = datetime.fromtimestamp(1777859410, tz=timezone.utc)

        fingerprint = _build_regime_fingerprint(
            market_slug="btc-updown-5m-1777859400",
            observed_at=observed_at,
            features=features,
            period_open_price_to_beat=78000.0,
        )

        self.assertIn(fingerprint["trend_regime"], {"weak_up", "strong_up"})

    def test_append_completed_order_tick_active_includes_feature_and_quote_data(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1777513811",
            market_title="Bitcoin Up or Down",
            side="UP",
            shares=5.0,
            entry_price=0.45,
            token_id="up-token",
            target_btc_price=77763.01,
            entry_btc_price=77760.00,
        )
        features = SimpleNamespace(
            price_usd=77959.60,
            delta_from_previous_tick=0.0,
            momentum_1m=-12.5,
            momentum_5m=-6.079999999987194,
            velocity_15s=-3.2,
            velocity_30s=-7.4,
            momentum_acceleration=4.2,
            volatility_5m=7.832826962356144,
            consecutive_flat_ticks=2,
            consecutive_directional_ticks=4,
            window_open_price=77970.0,
            delta_pct_from_window_open=-0.000133,
            trailing_5m_open_price=77965.0,
            delta_pct_from_trailing_5m_open=-0.000069,
            rsi_14=48.2,
        )
        up_snapshot = SimpleNamespace(
            buy_quote=0.25,
            reference_price=0.24,
            target_limit_price=0.24,
            recommended_limit_price=0.24,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.23,
            best_ask=0.25,
            spread=0.02,
        )
        down_snapshot = SimpleNamespace(
            buy_quote=0.70,
            reference_price=0.69,
            target_limit_price=0.69,
            recommended_limit_price=0.69,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.68,
            best_ask=0.70,
            spread=0.02,
        )

        with patch("custom.btc_agent.main.os.getcwd", return_value="/appl/agents"):
            append_completed_order_tick(
                order,
                current_btc_price=77959.60,
                phase="ACTIVE",
                features=features,
                up_snapshot=up_snapshot,
                down_snapshot=down_snapshot,
            )

        with open(
            "/appl/agents/completed_orders/completed_order_1777513811.txt",
            encoding="utf-8",
        ) as order_file:
            content = order_file.read()

        self.assertIn("phase=ACTIVE", content)
        self.assertIn("feature_btc_price=77959.600", content)
        self.assertIn("market_time_remaining_mmss=", content)
        self.assertIn("feature_momentum_1m=-12.5", content)
        self.assertIn("feature_momentum_5m=-6.079999999987194", content)
        self.assertIn("feature_velocity_15s=-3.2", content)
        self.assertIn("feature_velocity_30s=-7.4", content)
        self.assertIn("feature_momentum_acceleration=4.2", content)
        self.assertIn("feature_volatility_5m=7.832826962356144", content)
        self.assertIn("feature_consecutive_flat_ticks=2", content)
        self.assertIn("feature_consecutive_directional_ticks=4", content)
        self.assertIn("feature_last_10_ticks_direction=", content)
        self.assertIn("active_up_buy_quote=0.250", content)
        self.assertIn("active_down_buy_quote=0.700", content)
        self.assertIn("\"liquidity_regime\"", content)
        self.assertIn("\"next_slug_proximity\"", content)
        self.assertIn("\"imbalance_pressure\"", content)

    def test_append_completed_order_tick_uses_trade_number_suffix_when_multiple_trades_allowed(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1777675200",
            market_title="Bitcoin Up or Down",
            side="UP",
            shares=5.0,
            entry_price=0.45,
            token_id="up-token",
            target_btc_price=77763.01,
            entry_btc_price=77760.00,
            trade_number_in_period=2,
        )

        with patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/appl/agents",
        ), patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(max_trades_per_period=2),
        ):
            append_completed_order_tick(order, current_btc_price=77770.0, phase="PLACED")

        with open(
            "/appl/agents/completed_orders/completed_order_1777675200-2.txt",
            encoding="utf-8",
        ) as order_file:
            content = order_file.read()

        self.assertIn("phase=PLACED", content)

    def test_append_pending_period_tick_analysis_writes_pre_order_tick_data(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1999999999",
            title="Bitcoin Up or Down",
            settlement_threshold=77763.01,
        )
        up_snapshot = SimpleNamespace(
            buy_quote=0.45,
            reference_price=0.44,
            target_limit_price=0.44,
            recommended_limit_price=0.44,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.43,
            best_ask=0.45,
            spread=0.02,
        )
        down_snapshot = SimpleNamespace(
            buy_quote=0.52,
            reference_price=0.51,
            target_limit_price=0.51,
            recommended_limit_price=0.51,
            ok_to_submit=False,
            submit_reason="blocked",
            best_bid=0.50,
            best_ask=0.52,
            spread=0.02,
        )
        features = SimpleNamespace(
            price_usd=77770.0,
            delta_from_previous_tick=5.0,
            momentum_1m=7.0,
            momentum_5m=10.0,
            velocity_15s=3.0,
            velocity_30s=4.0,
            momentum_acceleration=-1.0,
            volatility_5m=22.0,
            consecutive_flat_ticks=0,
            consecutive_directional_ticks=3,
            window_open_price=77760.0,
            delta_pct_from_window_open=0.0001286,
            trailing_5m_open_price=77750.0,
            delta_pct_from_trailing_5m_open=0.0002572,
            rsi_14=55.0,
            as_of=SimpleNamespace(isoformat=lambda: "2026-05-01T01:00:00+00:00"),
        )
        decision = SimpleNamespace(
            side="UP",
            confidence=0.88,
            max_price_to_pay=1.0,
            reason="strong setup",
        )

        with patch("custom.btc_agent.main.os.getcwd", return_value="/appl/agents"):
            append_pending_period_tick_analysis(
                market,
                up_snapshot=up_snapshot,
                down_snapshot=down_snapshot,
                features=features,
                decision=decision,
                observed_at=SimpleNamespace(isoformat=lambda: "2026-05-01T01:00:00+00:00"),
            )

        with open(
            "/appl/agents/completed_orders/pending_period_1999999999.txt",
            encoding="utf-8",
        ) as analysis_file:
            content = analysis_file.read()

        self.assertIn("phase=PRE_ORDER_TICK", content)
        self.assertIn("up_buy_quote=0.450", content)
        self.assertIn("down_submit_reason=blocked", content)
        self.assertIn("decision_side=UP", content)
        self.assertIn("min_confidence=", content)
        self.assertIn("veto_threshold_delta=", content)
        self.assertIn("llm_raw_reasoning=", content)
        self.assertIn("velocity_15s=3.0", content)
        self.assertIn("momentum_alignment=", content)
        self.assertIn("consecutive_flat_ticks=0", content)
        self.assertIn("last_10_ticks_direction=", content)
        self.assertIn("market_time_remaining_mmss=", content)
        self.assertIn("\"next_slug_proximity\"", content)
        self.assertIn("\"required_velocity_to_win\"", content)
        self.assertIn("\"oracle_gap_ratio\"", content)
        self.assertIn("\"implied_oracle_price\"", content)
        self.assertIn("\"feed_drift_usd\"", content)

    def test_promote_pending_period_log_to_completed_copies_and_preserves_file(self):
        with open(
            "/appl/agents/completed_orders/pending_period_1999999998.txt",
            "w",
            encoding="utf-8",
        ) as pending_file:
            pending_file.write("pre-order analysis\n")

        completed_path = "/appl/agents/completed_orders/completed_order_1999999998.txt"
        if os.path.exists(completed_path):
            os.remove(completed_path)

        with patch("custom.btc_agent.main.os.getcwd", return_value="/appl/agents"):
            promote_pending_period_log_to_completed("btc-updown-5m-1999999998")

        with open(
            completed_path,
            encoding="utf-8",
        ) as completed_file:
            content = completed_file.read()

        self.assertEqual(content, "pre-order analysis\n")
        with open(
            "/appl/agents/completed_orders/pending_period_1999999998.txt",
            encoding="utf-8",
        ) as pending_file:
            self.assertEqual(pending_file.read(), "pre-order analysis\n")

    def test_finalize_pending_period_log_renames_unexecuted_period_analysis(self):
        with open(
            "/appl/agents/completed_orders/pending_period_1999999997.txt",
            "w",
            encoding="utf-8",
        ) as pending_file:
            pending_file.write("pre-order analysis\n")

        with patch("custom.btc_agent.main.os.getcwd", return_value="/appl/agents"):
            finalize_pending_period_log("btc-updown-5m-1999999997")

        self.assertFalse(os.path.exists("/appl/agents/completed_orders/pending_period_1999999997.txt"))
        with open(
            "/appl/agents/completed_orders/completed_period_1999999997.txt",
            encoding="utf-8",
        ) as completed_file:
            self.assertEqual(completed_file.read(), "pre-order analysis\n")

    def test_finalize_pending_period_log_uses_period_direction_suffix_when_final_price_known(self):
        with open(
            "/appl/agents/completed_orders/pending_period_1999999997.txt",
            "w",
            encoding="utf-8",
        ) as pending_file:
            pending_file.write("period_open_price_to_beat=77763.01\npre-order analysis\n")

        with patch("custom.btc_agent.main.os.getcwd", return_value="/appl/agents"):
            finalize_pending_period_log("btc-updown-5m-1999999997", final_btc_price=77780.0)

        self.assertFalse(os.path.exists("/appl/agents/completed_orders/pending_period_1999999997.txt"))
        with open(
            "/appl/agents/completed_orders/completed_period_up_1999999997.txt",
            encoding="utf-8",
        ) as completed_file:
            self.assertIn("pre-order analysis", completed_file.read())

    def test_finalize_current_period_logs_on_exit_renames_active_pending_period(self):
        with open(
            "/appl/agents/completed_orders/pending_period_1999999999.txt",
            "w",
            encoding="utf-8",
        ) as pending_file:
            pending_file.write("pre-order analysis\n")

        with patch(
            "custom.btc_agent.main.get_state",
            return_value=SimpleNamespace(market_slug="btc-updown-5m-1999999999"),
        ), patch("custom.btc_agent.main.os.getcwd", return_value="/appl/agents"):
            finalize_current_period_logs_on_exit()

        self.assertFalse(os.path.exists("/appl/agents/completed_orders/pending_period_1999999999.txt"))
        with open(
            "/appl/agents/completed_orders/completed_period_1999999999.txt",
            encoding="utf-8",
        ) as completed_file:
            self.assertEqual(completed_file.read(), "pre-order analysis\n")

    def test_write_price_to_beat_debug_file_writes_report(self):
        with patch(
            "custom.btc_agent.main.build_price_to_beat_debug_reports",
            return_value=["page debug report\n", "next data debug report\n", "third page debug report\n"],
        ) as mock_build_reports, patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/appl/agents",
        ), patch(
            "custom.btc_agent.main._DEBUG_WRITTEN_SLUGS",
            set(),
        ):
            write_price_to_beat_debug_file("btc-updown-5m-1776983100")

        mock_build_reports.assert_not_called()
        self.assertFalse(os.path.exists("/appl/agents/logs/priceToBeatDebug.txt"))
        self.assertFalse(os.path.exists("/appl/agents/logs/priceToBeatDebugPg2.txt"))
        self.assertFalse(os.path.exists("/appl/agents/logs/priceToBeatDebugPg3.txt"))

    def test_write_price_to_beat_debug_file_writes_only_once_per_slug_without_force(self):
        with patch(
            "custom.btc_agent.main.build_price_to_beat_debug_reports",
            return_value=["page debug report\n", "next data debug report\n"],
        ) as mock_build_reports, patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/appl/agents",
        ), patch(
            "custom.btc_agent.main._DEBUG_WRITTEN_SLUGS",
            set(),
        ):
            write_price_to_beat_debug_file("btc-updown-5m-1776983100")
            write_price_to_beat_debug_file("btc-updown-5m-1776983100")

        mock_build_reports.assert_not_called()

    def test_clear_price_to_beat_debug_files_removes_only_price_to_beat_logs(self):
        with patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/appl/agents",
        ), patch(
            "custom.btc_agent.main.os.listdir",
            return_value=[
                "priceToBeatDebug.txt",
                "priceToBeatDebugPg2.txt",
                "unrelated.log",
            ],
        ), patch("custom.btc_agent.main.os.remove") as mock_remove:
            clear_price_to_beat_debug_files()

        mock_remove.assert_not_called()

    def test_resolve_price_to_beat_with_retries_refreshes_same_slug(self):
        initial_market = SimpleNamespace(slug="btc-updown-5m-1777056000", settlement_threshold=None)
        refreshed_market = SimpleNamespace(
            slug="btc-updown-5m-1777056000",
            settlement_threshold=77560.75,
        )

        with patch(
            "custom.btc_agent.main.get_btc_updown_market_by_slug",
            return_value=refreshed_market,
        ) as mock_get_by_slug, patch(
            "custom.btc_agent.main.time.sleep",
        ):
            market = resolve_price_to_beat_with_retries(initial_market, retry_attempts=2, retry_delay_seconds=1)

        self.assertEqual(market.settlement_threshold, 77560.75)
        mock_get_by_slug.assert_called_once_with("btc-updown-5m-1777056000")

    def test_resolve_price_to_beat_with_retries_defaults_to_twelve_attempts(self):
        initial_market = SimpleNamespace(slug="btc-updown-5m-1777056000", settlement_threshold=None)

        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(debug=False),
        ), patch(
            "custom.btc_agent.main.get_btc_updown_market_by_slug",
            return_value=None,
        ) as mock_get_by_slug, patch(
            "custom.btc_agent.main.time.sleep",
        ):
            market = resolve_price_to_beat_with_retries(initial_market, retry_delay_seconds=0)

        self.assertIs(market, initial_market)
        self.assertEqual(mock_get_by_slug.call_count, 12)

    def test_resolve_price_to_beat_with_retries_writes_debug_attempts_to_pending_period_log(self):
        initial_market = SimpleNamespace(slug="btc-updown-5m-1777056000", settlement_threshold=None)
        refreshed_market = SimpleNamespace(
            slug="btc-updown-5m-1777056000",
            settlement_threshold=77560.75,
        )
        log_path = "/tmp/completed_orders/pending_period_1777056000.txt"

        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(debug=True),
        ), patch(
            "custom.btc_agent.main.get_btc_updown_market_by_slug",
            return_value=refreshed_market,
        ), patch(
            "custom.btc_agent.main.time.sleep",
        ), patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/tmp",
        ):
            market = resolve_price_to_beat_with_retries(initial_market, retry_attempts=1, retry_delay_seconds=0)

        self.assertEqual(market.settlement_threshold, 77560.75)
        with open(log_path, encoding="utf-8") as pending_log:
            content = pending_log.read()
        self.assertIn("price_to_beat_retry: attempt 1/1 for btc-updown-5m-1777056000 after 0s", content)
        self.assertIn("[DEBUG] Retry 1 resulted in threshold: 77560.75", content)

    def test_resolve_price_to_beat_with_retries_skips_retries_when_debug_price_to_beat_enabled(self):
        initial_market = SimpleNamespace(slug="btc-updown-5m-1777056000", settlement_threshold=None)
        refreshed_market = SimpleNamespace(
            slug="btc-updown-5m-1777056000",
            settlement_threshold=77560.75,
        )

        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(debug_price_to_beat=True),
        ), patch(
            "custom.btc_agent.main.get_btc_updown_market_by_slug",
            return_value=refreshed_market,
        ) as mock_get_by_slug, patch(
            "custom.btc_agent.main.time.sleep",
        ):
            market = resolve_price_to_beat_with_retries(initial_market, retry_attempts=2, retry_delay_seconds=1)

        self.assertEqual(market.settlement_threshold, 77560.75)
        mock_get_by_slug.assert_called_once_with("btc-updown-5m-1777056000")

    def test_run_once_writes_price_to_beat_debug_file_when_debug_enabled(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1777056000",
            title="Bitcoin Up or Down",
            settlement_threshold=77560.75,
            up_token_id="up-token",
            down_token_id="down-token",
        )
        state = SimpleNamespace(trades_executed=1)

        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(
                debug=True,
                debug_price_to_beat=False,
                max_trades_per_period=1,
                minimum_wallet_balance=0.0,
            ),
        ), patch(
            "custom.btc_agent.main.find_current_btc_updown_market",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.sync_period_state",
            return_value=False,
        ), patch(
            "custom.btc_agent.main.get_state",
            return_value=state,
        ), patch(
            "custom.btc_agent.main.resolve_price_to_beat_with_retries",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.write_price_to_beat_debug_file",
        ) as mock_write_debug, patch(
            "custom.btc_agent.main.fetch_btc_spot_price",
            return_value=77560.75,
        ), patch(
            "custom.btc_agent.main.print_active_orders",
        ), patch(
            "custom.btc_agent.main.print_account_snapshot_from_snapshot",
        ), patch(
            "custom.btc_agent.main.get_account_balance_snapshot",
            return_value=SimpleNamespace(cash_balance=100.0),
        ), patch(
            "custom.btc_agent.main.enforce_minimum_wallet_balance",
        ):
            run_once()

        mock_write_debug.assert_not_called()

    def test_run_once_clears_price_to_beat_debug_files_on_new_slug(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1777056000",
            title="Bitcoin Up or Down",
            settlement_threshold=77560.75,
            up_token_id="up-token",
            down_token_id="down-token",
        )
        state = SimpleNamespace(trades_executed=1)

        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(
                debug=True,
                debug_price_to_beat=False,
                max_trades_per_period=1,
                minimum_wallet_balance=0.0,
            ),
        ), patch(
            "custom.btc_agent.main.find_current_btc_updown_market",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.sync_period_state",
            return_value=True,
        ), patch(
            "custom.btc_agent.main.get_state",
            return_value=state,
        ), patch(
            "custom.btc_agent.main.resolve_price_to_beat_with_retries",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.write_price_to_beat_debug_file",
        ), patch(
            "custom.btc_agent.main.clear_price_to_beat_debug_files",
        ) as mock_clear_debug, patch(
            "custom.btc_agent.main.fetch_btc_spot_price",
            return_value=77560.75,
        ), patch(
            "custom.btc_agent.main.print_active_orders",
        ), patch(
            "custom.btc_agent.main.print_account_snapshot_from_snapshot",
        ), patch(
            "custom.btc_agent.main.get_account_balance_snapshot",
            return_value=SimpleNamespace(cash_balance=100.0),
        ), patch(
            "custom.btc_agent.main.enforce_minimum_wallet_balance",
        ), patch(
            "custom.btc_agent.main._DEBUG_WRITTEN_SLUGS",
            {"old-slug"},
        ):
            run_once()

        mock_clear_debug.assert_not_called()

    def test_run_once_finalizes_no_trade_previous_period_with_directional_price(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1777056300",
            title="Bitcoin Up or Down",
            settlement_threshold=77560.75,
            up_token_id="up-token",
            down_token_id="down-token",
        )
        previous_state = SimpleNamespace(
            active_orders=[],
            market_slug="btc-updown-5m-1777056000",
            trades_executed=1,
        )

        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(
                debug=False,
                debug_price_to_beat=False,
                max_trades_per_period=1,
                minimum_wallet_balance=0.0,
                llm_connection_debug=False,
            ),
        ), patch(
            "custom.btc_agent.main.find_current_btc_updown_market",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.sync_period_state",
            return_value=True,
        ), patch(
            "custom.btc_agent.main.get_state",
            side_effect=[previous_state, previous_state],
        ), patch(
            "custom.btc_agent.main.resolve_price_to_beat_with_retries",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.finalize_pending_period_log",
        ) as mock_finalize_period, patch(
            "custom.btc_agent.main.get_account_balance_snapshot",
            return_value=SimpleNamespace(cash_balance=100.0),
        ), patch(
            "custom.btc_agent.main.print_account_snapshot_from_snapshot",
        ), patch(
            "custom.btc_agent.main.enforce_minimum_wallet_balance",
        ), patch(
            "custom.btc_agent.main.get_feature_readiness",
            return_value=(False, "not ready"),
        ), patch(
            "custom.btc_agent.main.clear_price_to_beat_debug_files",
        ), patch(
            "custom.btc_agent.main.fetch_btc_spot_price",
            return_value=77560.75,
        ):
            run_once()

        mock_finalize_period.assert_called_once_with(
            "btc-updown-5m-1777056000",
            77560.75,
        )

    def test_run_once_does_not_exit_after_executed_trade_when_no_loss_is_recorded_yet(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1777056000",
            title="Bitcoin Up or Down",
            settlement_threshold=77560.75,
            up_token_id="up-token",
            down_token_id="down-token",
            start_ts=1777056000,
        )
        state = SimpleNamespace(trades_executed=0)
        features = SimpleNamespace(
            price_usd=77560.75,
            as_of=None,
            delta_from_previous_tick=1.0,
            momentum_1m=1.0,
            momentum_5m=2.0,
            velocity_15s=0.5,
            velocity_30s=1.0,
            momentum_acceleration=-0.5,
            volatility_5m=3.0,
            consecutive_flat_ticks=0,
            consecutive_directional_ticks=2,
        )
        decision = SimpleNamespace(
            side="UP",
            confidence=0.8,
            max_price_to_pay=0.5,
            reason="test",
        )
        result = SimpleNamespace(
            executed=True,
            side="UP",
            size=5.0,
            price=0.45,
            token_id="up-token",
            reason="executed",
            execution_snapshot=None,
        )

        with ExitStack() as stack:
            stack.enter_context(patch("custom.btc_agent.main._FIRST_LOOP", False))
            stack.enter_context(patch("custom.btc_agent.main._SESSION_LOSS_TRADES", 0))
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_trading_config",
                    return_value=SimpleNamespace(
                        debug=False,
                        debug_price_to_beat=False,
                        max_trades_per_period=1,
                        max_losses_per_run=1,
                        minimum_wallet_balance=0.0,
                    ),
                )
            )
            stack.enter_context(
                patch("custom.btc_agent.main.find_current_btc_updown_market", return_value=market)
            )
            stack.enter_context(patch("custom.btc_agent.main.sync_period_state", return_value=False))
            stack.enter_context(patch("custom.btc_agent.main.get_state", return_value=state))
            stack.enter_context(
                patch("custom.btc_agent.main.resolve_price_to_beat_with_retries", return_value=market)
            )
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_token_quote_snapshot",
                    side_effect=[
                        SimpleNamespace(ok_to_submit=True),
                        SimpleNamespace(ok_to_submit=True),
                    ],
                )
            )
            stack.enter_context(
                patch("custom.btc_agent.main.build_btc_features", return_value=features)
            )
            stack.enter_context(
                patch("custom.btc_agent.main.get_feature_readiness", return_value=(True, None))
            )
            stack.enter_context(patch("custom.btc_agent.main.decide_trade", return_value=decision))
            stack.enter_context(
                patch("custom.btc_agent.main.get_decision_quote_snapshot", return_value=SimpleNamespace())
            )
            stack.enter_context(
                patch("custom.btc_agent.main.maybe_execute_trade", return_value=result)
            )
            stack.enter_context(patch("custom.btc_agent.main.print_account_snapshot_from_snapshot"))
            stack.enter_context(patch("custom.btc_agent.main.enforce_minimum_wallet_balance"))
            stack.enter_context(patch("custom.btc_agent.main.print_market_context"))
            stack.enter_context(patch("custom.btc_agent.main.print_quote_snapshot_from_snapshot"))
            stack.enter_context(patch("custom.btc_agent.main.print_features"))
            stack.enter_context(patch("custom.btc_agent.main.print_llm_decision"))
            stack.enter_context(patch("custom.btc_agent.main.print_trade_execution_result"))
            stack.enter_context(patch("custom.btc_agent.main.record_executed_trade"))
            stack.enter_context(patch("custom.btc_agent.main.get_active_orders", return_value=[]))
            mock_exit = stack.enter_context(
                patch("custom.btc_agent.main.sys.exit")
            )
            run_once()

        mock_exit.assert_not_called()

    def test_run_once_collects_internal_quote_context_without_printing_snapshots_when_recommended_limit_disabled(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1777056000",
            title="Bitcoin Up or Down",
            settlement_threshold=77560.75,
            up_token_id="up-token",
            down_token_id="down-token",
            start_ts=1777056000,
        )
        state = SimpleNamespace(trades_executed=0)
        features = SimpleNamespace(
            price_usd=77560.75,
            as_of=SimpleNamespace(isoformat=lambda: "2026-05-01T01:00:00+00:00"),
            delta_from_previous_tick=1.0,
            momentum_1m=1.0,
            momentum_5m=2.0,
            velocity_15s=0.5,
            velocity_30s=1.0,
            momentum_acceleration=-0.5,
            volatility_5m=3.0,
            consecutive_flat_ticks=0,
            consecutive_directional_ticks=2,
        )
        decision = SimpleNamespace(
            side="UP",
            confidence=0.8,
            max_price_to_pay=1.0,
            reason="test",
        )
        decision_snapshot = SimpleNamespace(
            token_id="up-token",
            buy_quote=0.45,
            midpoint=0.45,
            last_trade_price=0.45,
            reference_price=0.45,
            target_limit_price=0.45,
            recommended_limit_price=0.45,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.44,
            best_ask=0.45,
            tick_size=0.01,
            spread=0.01,
        )
        result = SimpleNamespace(
            executed=False,
            side="UP",
            size=0.0,
            price=0.45,
            token_id="up-token",
            reason="not executed",
            execution_snapshot=decision_snapshot,
        )

        with ExitStack() as stack:
            stack.enter_context(patch("custom.btc_agent.main._FIRST_LOOP", False))
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_trading_config",
                    return_value=SimpleNamespace(
                        debug=False,
                        debug_price_to_beat=False,
                        use_recommended_limit=False,
                        max_trades_per_period=1,
                        max_losses_per_run=0,
                        minimum_wallet_balance=0.0,
                    ),
                )
            )
            stack.enter_context(
                patch("custom.btc_agent.main.find_current_btc_updown_market", return_value=market)
            )
            stack.enter_context(patch("custom.btc_agent.main.sync_period_state", return_value=False))
            stack.enter_context(patch("custom.btc_agent.main.get_state", return_value=state))
            stack.enter_context(
                patch("custom.btc_agent.main.resolve_price_to_beat_with_retries", return_value=market)
            )
            mock_get_token_quote_snapshot = stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_token_quote_snapshot",
                    return_value=decision_snapshot,
                )
            )
            stack.enter_context(
                patch("custom.btc_agent.main.build_btc_features", return_value=features)
            )
            stack.enter_context(
                patch("custom.btc_agent.main.get_feature_readiness", return_value=(True, None))
            )
            stack.enter_context(patch("custom.btc_agent.main.decide_trade", return_value=decision))
            mock_print_quote_snapshot = stack.enter_context(
                patch("custom.btc_agent.main.print_quote_snapshot_from_snapshot")
            )
            stack.enter_context(
                patch("custom.btc_agent.main.maybe_execute_trade", return_value=result)
            )
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_account_balance_snapshot",
                    return_value=SimpleNamespace(cash_balance=100.0, total_account_value=100.0),
                )
            )
            stack.enter_context(patch("custom.btc_agent.main.print_market_context"))
            stack.enter_context(patch("custom.btc_agent.main.print_features"))
            stack.enter_context(patch("custom.btc_agent.main.print_llm_decision"))
            stack.enter_context(patch("custom.btc_agent.main.print_trade_execution_result"))
            stack.enter_context(patch("custom.btc_agent.main.get_active_orders", return_value=[]))

            run_once()

        self.assertEqual(mock_get_token_quote_snapshot.call_count, 3)
        self.assertEqual(mock_get_token_quote_snapshot.call_args_list[0].args, ("up-token",))
        self.assertEqual(mock_get_token_quote_snapshot.call_args_list[0].kwargs, {"market": market})
        self.assertEqual(mock_get_token_quote_snapshot.call_args_list[1].args, ("down-token",))
        self.assertEqual(mock_get_token_quote_snapshot.call_args_list[1].kwargs, {"market": market})
        self.assertEqual(mock_get_token_quote_snapshot.call_args_list[2].args, ("up-token",))
        self.assertEqual(mock_get_token_quote_snapshot.call_args_list[2].kwargs, {"decision": decision, "market": market})
        mock_print_quote_snapshot.assert_not_called()

    def test_run_once_skips_new_trade_during_post_execution_cooldown(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1777056000",
            title="Bitcoin Up or Down",
            settlement_threshold=77560.75,
            up_token_id="up-token",
            down_token_id="down-token",
            start_ts=1777056000,
        )
        state = SimpleNamespace(trades_executed=1)
        features = SimpleNamespace(
            price_usd=77560.75,
            as_of=None,
            delta_from_previous_tick=1.0,
            momentum_1m=1.0,
            momentum_5m=2.0,
            velocity_15s=0.5,
            velocity_30s=1.0,
            momentum_acceleration=-0.5,
            volatility_5m=3.0,
            consecutive_flat_ticks=0,
            consecutive_directional_ticks=2,
        )

        with ExitStack() as stack:
            stack.enter_context(patch("custom.btc_agent.main._FIRST_LOOP", False))
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_trading_config",
                    return_value=SimpleNamespace(
                        debug=False,
                        debug_price_to_beat=False,
                        use_recommended_limit=False,
                        max_trades_per_period=2,
                        max_losses_per_run=0,
                        minimum_wallet_balance=0.0,
                    ),
                )
            )
            stack.enter_context(
                patch("custom.btc_agent.main.find_current_btc_updown_market", return_value=market)
            )
            stack.enter_context(patch("custom.btc_agent.main.sync_period_state", return_value=False))
            stack.enter_context(patch("custom.btc_agent.main.get_state", return_value=state))
            stack.enter_context(
                patch("custom.btc_agent.main.resolve_price_to_beat_with_retries", return_value=market)
            )
            stack.enter_context(
                patch("custom.btc_agent.main.get_trade_cooldown_remaining", return_value=3)
            )
            mock_consume_cooldown = stack.enter_context(
                patch("custom.btc_agent.main.consume_trade_cooldown_loop")
            )
            stack.enter_context(
                patch("custom.btc_agent.main.build_btc_features", return_value=features)
            )
            stack.enter_context(
                patch("custom.btc_agent.main.get_active_orders", return_value=[])
            )
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_account_balance_snapshot",
                    return_value=SimpleNamespace(cash_balance=100.0, total_account_value=100.0),
                )
            )
            stack.enter_context(patch("custom.btc_agent.main.print_market_context"))
            stack.enter_context(patch("custom.btc_agent.main.print_features"))
            mock_print_skip = stack.enter_context(
                patch("custom.btc_agent.main.print_llm_skip_reason")
            )
            mock_decide_trade = stack.enter_context(
                patch("custom.btc_agent.main.decide_trade")
            )
            mock_execute_trade = stack.enter_context(
                patch("custom.btc_agent.main.maybe_execute_trade")
            )
            run_once()

        mock_consume_cooldown.assert_called_once()
        mock_print_skip.assert_called_once()
        mock_decide_trade.assert_not_called()
        mock_execute_trade.assert_not_called()

    def test_run_once_skips_new_trade_when_existing_active_order_is_losing(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1777056000",
            title="Bitcoin Up or Down",
            settlement_threshold=77560.75,
            up_token_id="up-token",
            down_token_id="down-token",
            start_ts=1777056000,
        )
        state = SimpleNamespace(trades_executed=1)
        features = SimpleNamespace(
            price_usd=77550.0,
            as_of=None,
            delta_from_previous_tick=-2.0,
            momentum_1m=-1.0,
            momentum_5m=-3.0,
            velocity_15s=-1.0,
            velocity_30s=-2.0,
            momentum_acceleration=1.0,
            volatility_5m=4.0,
            consecutive_flat_ticks=0,
            consecutive_directional_ticks=6,
            delta_pct_from_window_open=-0.0002,
            rsi_14=45.0,
        )
        active_order = ActivePaperOrder(
            market_slug="btc-updown-5m-1777056000",
            market_title="Bitcoin Up or Down",
            side="UP",
            shares=2.0,
            entry_price=0.45,
            token_id="up-token",
            target_btc_price=77560.75,
            entry_btc_price=77570.0,
            actual_fill_price=0.45,
            filled=True,
        )

        with ExitStack() as stack:
            stack.enter_context(patch("custom.btc_agent.main._FIRST_LOOP", False))
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_trading_config",
                    return_value=SimpleNamespace(
                        debug=False,
                        debug_price_to_beat=False,
                        use_recommended_limit=False,
                        max_trades_per_period=2,
                        max_losses_per_run=0,
                        minimum_wallet_balance=0.0,
                    ),
                )
            )
            stack.enter_context(
                patch("custom.btc_agent.main.find_current_btc_updown_market", return_value=market)
            )
            stack.enter_context(patch("custom.btc_agent.main.sync_period_state", return_value=False))
            stack.enter_context(patch("custom.btc_agent.main.get_state", return_value=state))
            stack.enter_context(
                patch("custom.btc_agent.main.resolve_price_to_beat_with_retries", return_value=market)
            )
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_account_balance_snapshot",
                    return_value=SimpleNamespace(cash_balance=100.0, total_account_value=100.0),
                )
            )
            stack.enter_context(
                patch("custom.btc_agent.main.build_btc_features", return_value=features)
            )
            stack.enter_context(
                patch("custom.btc_agent.main.get_feature_readiness", return_value=(True, None))
            )
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_token_quote_snapshot",
                    return_value=SimpleNamespace(
                        buy_quote=0.45,
                        reference_price=0.45,
                        target_limit_price=0.45,
                        recommended_limit_price=0.45,
                        ok_to_submit=True,
                        submit_reason="ok",
                        best_bid=0.44,
                        best_ask=0.45,
                        best_bid_size=100.0,
                        best_ask_size=80.0,
                        spread=0.01,
                        spread_bps=22.0,
                        top_level_book_imbalance=0.56,
                        imbalance_pressure=0.12,
                    ),
                )
            )
            stack.enter_context(
                patch("custom.btc_agent.main.get_active_orders", return_value=[active_order])
            )
            stack.enter_context(patch("custom.btc_agent.main.print_market_context"))
            stack.enter_context(patch("custom.btc_agent.main.print_features"))
            mock_update_logs = stack.enter_context(
                patch("custom.btc_agent.main.update_active_order_logs")
            )
            mock_print_active = stack.enter_context(
                patch("custom.btc_agent.main.print_active_orders")
            )
            mock_print_skip = stack.enter_context(
                patch("custom.btc_agent.main.print_llm_skip_reason")
            )
            mock_decide_trade = stack.enter_context(
                patch("custom.btc_agent.main.decide_trade")
            )
            mock_execute_trade = stack.enter_context(
                patch("custom.btc_agent.main.maybe_execute_trade")
            )

            run_once()

        mock_update_logs.assert_called_once()
        mock_print_active.assert_called_once_with(77550.0, observed_at=None)
        mock_print_skip.assert_called_once()
        mock_decide_trade.assert_not_called()
        mock_execute_trade.assert_not_called()

    def test_run_once_hard_returns_when_existing_active_order_is_unfilled(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1777056000",
            title="Bitcoin Up or Down",
            settlement_threshold=77560.75,
            up_token_id="up-token",
            down_token_id="down-token",
            start_ts=1777056000,
        )
        state = SimpleNamespace(trades_executed=0, active_orders=[])
        features = SimpleNamespace(
            price_usd=77550.0,
            as_of=datetime.fromtimestamp(1777056100, tz=timezone.utc),
        )
        active_order = ActivePaperOrder(
            market_slug="btc-updown-5m-1777056000",
            market_title="Bitcoin Up or Down",
            side="UP",
            shares=2.0,
            entry_price=0.45,
            token_id="up-token",
            target_btc_price=77560.75,
            entry_btc_price=77570.0,
            live_order_id="live-order-1",
            filled=False,
            api_order_state="ORDER_STATE_NEW",
        )

        with ExitStack() as stack:
            stack.enter_context(patch("custom.btc_agent.main._FIRST_LOOP", False))
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_trading_config",
                    return_value=SimpleNamespace(
                        debug=False,
                        debug_price_to_beat=False,
                        use_recommended_limit=False,
                        max_trades_per_period=2,
                        max_losses_per_run=0,
                        minimum_wallet_balance=0.0,
                    ),
                )
            )
            stack.enter_context(
                patch("custom.btc_agent.main.find_current_btc_updown_market", return_value=market)
            )
            stack.enter_context(patch("custom.btc_agent.main.sync_period_state", return_value=False))
            stack.enter_context(patch("custom.btc_agent.main.get_state", return_value=state))
            stack.enter_context(
                patch("custom.btc_agent.main.resolve_price_to_beat_with_retries", return_value=market)
            )
            stack.enter_context(
                patch("custom.btc_agent.main.build_btc_features", return_value=features)
            )
            stack.enter_context(
                patch("custom.btc_agent.main.get_feature_readiness", return_value=(True, None))
            )
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_token_quote_snapshot",
                    return_value=SimpleNamespace(
                        buy_quote=0.45,
                        ok_to_submit=True,
                        submit_reason="ok",
                    ),
                )
            )
            stack.enter_context(
                patch("custom.btc_agent.main.get_active_orders", return_value=[active_order])
            )
            mock_refresh = stack.enter_context(
                patch("custom.btc_agent.main.refresh_live_order_fill_status", return_value=False)
            )
            stack.enter_context(patch("custom.btc_agent.main.print_market_context"))
            stack.enter_context(patch("custom.btc_agent.main.print_features"))
            mock_update_logs = stack.enter_context(
                patch("custom.btc_agent.main.update_active_order_logs")
            )
            mock_print_active = stack.enter_context(
                patch("custom.btc_agent.main.print_active_orders")
            )
            mock_print = stack.enter_context(patch("builtins.print"))
            mock_decide_trade = stack.enter_context(
                patch("custom.btc_agent.main.decide_trade")
            )
            mock_execute_trade = stack.enter_context(
                patch("custom.btc_agent.main.maybe_execute_trade")
            )

            run_once()

        mock_refresh.assert_called_once_with(active_order)
        mock_update_logs.assert_called_once()
        mock_print_active.assert_called_once_with(77550.0, observed_at=features.as_of)
        self.assertTrue(
            any("is UNFILLED. Waiting for fill or period to end" in str(call.args[0]) for call in mock_print.call_args_list)
        )
        mock_decide_trade.assert_not_called()
        mock_execute_trade.assert_not_called()

    def test_run_once_finalizes_previous_orders_on_new_slug(self):
        previous_order = ActivePaperOrder(
            market_slug="btc-updown-5m-1777513500",
            market_title="Prior Period",
            side="DOWN",
            shares=5.0,
            entry_price=0.43,
            token_id="down-token",
            target_btc_price=77720.0,
            entry_btc_price=77725.0,
            actual_fill_price=0.43,
            filled=True,
        )
        market = SimpleNamespace(
            slug="btc-updown-5m-1777513800",
            title="Bitcoin Up or Down",
            settlement_threshold=77710.0,
            up_token_id="up-token",
            down_token_id="down-token",
        )
        previous_state = SimpleNamespace(active_orders=[previous_order])
        current_state = SimpleNamespace(trades_executed=1, active_orders=[])

        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(
                debug=False,
                debug_price_to_beat=False,
                max_trades_per_period=1,
                max_losses_per_run=0,
                minimum_wallet_balance=0.0,
            ),
        ), patch(
            "custom.btc_agent.main.find_current_btc_updown_market",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.get_state",
            side_effect=[previous_state, current_state],
        ), patch(
            "custom.btc_agent.main.sync_period_state",
            return_value=True,
        ), patch(
            "custom.btc_agent.main.resolve_price_to_beat_with_retries",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.fetch_btc_resolution_price_for_slug",
            return_value=77710.0,
        ), patch(
            "custom.btc_agent.main.get_account_balance_snapshot",
            return_value=SimpleNamespace(cash_balance=100.0),
        ), patch(
            "custom.btc_agent.main.print_account_snapshot_from_snapshot",
        ), patch(
            "custom.btc_agent.main.enforce_minimum_wallet_balance",
        ), patch(
            "custom.btc_agent.main.fetch_btc_spot_price",
            return_value=77710.0,
        ), patch(
            "custom.btc_agent.main.print_active_orders",
        ), patch(
            "custom.btc_agent.main.clear_price_to_beat_debug_files",
        ), patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/appl/agents",
        ):
            run_once()

        with open(
            "/appl/agents/completed_orders/completed_order_win_down_1777513500.txt",
            encoding="utf-8",
        ) as order_file:
            content = order_file.read()

        self.assertIn("phase=COMPLETED", content)
        self.assertIn("current_btc_price=77710.00", content)
        self.assertIn("outcome_label=win", content)

    def test_run_once_finalizes_previous_not_filled_orders_with_neutral_unfilled_filename(self):
        previous_order = ActivePaperOrder(
            market_slug="btc-updown-5m-1777513500",
            market_title="Prior Period",
            side="DOWN",
            shares=5.0,
            entry_price=0.43,
            token_id="down-token",
            target_btc_price=77720.0,
            entry_btc_price=77725.0,
            actual_fill_price=None,
        )
        market = SimpleNamespace(
            slug="btc-updown-5m-1777513800",
            title="Bitcoin Up or Down",
            settlement_threshold=77710.0,
            up_token_id="up-token",
            down_token_id="down-token",
        )
        previous_state = SimpleNamespace(active_orders=[previous_order])
        current_state = SimpleNamespace(trades_executed=1, active_orders=[])

        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(
                debug=False,
                debug_price_to_beat=False,
                max_trades_per_period=1,
                max_losses_per_run=0,
                minimum_wallet_balance=0.0,
            ),
        ), patch(
            "custom.btc_agent.main.find_current_btc_updown_market",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.get_state",
            side_effect=[previous_state, current_state],
        ), patch(
            "custom.btc_agent.main.sync_period_state",
            return_value=True,
        ), patch(
            "custom.btc_agent.main.resolve_price_to_beat_with_retries",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.fetch_btc_resolution_price_for_slug",
            return_value=77710.0,
        ), patch(
            "custom.btc_agent.main.get_account_balance_snapshot",
            return_value=SimpleNamespace(cash_balance=100.0),
        ), patch(
            "custom.btc_agent.main.print_account_snapshot_from_snapshot",
        ), patch(
            "custom.btc_agent.main.enforce_minimum_wallet_balance",
        ), patch(
            "custom.btc_agent.main.fetch_btc_spot_price",
            return_value=77710.0,
        ), patch(
            "custom.btc_agent.main.print_active_orders",
        ), patch(
            "custom.btc_agent.main.clear_price_to_beat_debug_files",
        ), patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/appl/agents",
        ):
            run_once()

        self.assertTrue(
            os.path.exists(
                "/appl/agents/completed_orders/completed_order_unfilled_down_1777513500.txt"
            )
        )
        with open(
            "/appl/agents/completed_orders/completed_order_unfilled_down_1777513500.txt",
            encoding="utf-8",
        ) as order_file:
            content = order_file.read()
        self.assertIn("outcome_label=unfilled", content)
        self.assertIn("position_state=UNFILLED", content)
        self.assertIn("filled=False", content)
        self.assertIn("internal_filled_flag=False", content)

    def test_run_once_tracks_live_submitted_but_unfilled_order_without_creating_win_file(self):
        market = SimpleNamespace(
            slug="btc-updown-5m-1778208300",
            title="Bitcoin Up or Down",
            settlement_threshold=80100.0,
            up_token_id="up-token",
            down_token_id="down-token",
            start_ts=1778208300,
        )
        state = SimpleNamespace(trades_executed=0, active_orders=[])
        features = SimpleNamespace(
            price_usd=80120.0,
            as_of=datetime.now(timezone.utc),
            delta_from_previous_tick=1.0,
            momentum_1m=2.0,
            momentum_5m=5.0,
            velocity_15s=1.0,
            velocity_30s=2.0,
            momentum_acceleration=-1.0,
            volatility_5m=10.0,
            consecutive_flat_ticks=0,
            consecutive_directional_ticks=3,
            last_10_ticks_direction="UUUD",
            delta_pct_from_window_open=0.001,
            rsi_9=60.0,
            rsi_14=55.0,
            rsi_speed_divergence=5.0,
            ema_9=80110.0,
            ema_21=80100.0,
            ema_alignment=True,
            ema_cross_direction="bullish",
            adx_14=35.0,
            atr_14=8.0,
            window_open_price=80100.0,
            trailing_window_open_price=80095.0,
        )
        up_snapshot = SimpleNamespace(
            token_id="up-token",
            buy_quote=0.67,
            reference_price=0.67,
            target_limit_price=0.67,
            recommended_limit_price=0.67,
            ok_to_submit=True,
            submit_reason="ok",
            best_bid=0.66,
            best_ask=0.67,
            best_bid_size=100.0,
            best_ask_size=100.0,
            spread=0.01,
            spread_bps=15.0,
            top_level_book_imbalance=0.5,
            imbalance_pressure=0.0,
        )
        decision = SimpleNamespace(
            side="UP",
            confidence=0.72,
            max_price_to_pay=1.0,
            reason="test reason",
            prompt_text=None,
            raw_response_text='{"decision":"UP"}',
        )
        result = SimpleNamespace(
            executed=False,
            submission_accepted=True,
            side="UP",
            size=5.0,
            price=0.67,
            token_id="up-token",
            reason="Live order submission was accepted but no fill was confirmed",
            execution_snapshot=up_snapshot,
            quoted_price_at_entry=0.67,
            actual_fill_price=None,
            realized_slippage_bps=None,
            order_latency_ms=250,
            book_depth_at_fill=100.0,
            shares_requested=5.0,
            live_order_id="order-123",
            api_order_state="ORDER_STATE_NEW",
        )

        with ExitStack() as stack:
            stack.enter_context(patch("custom.btc_agent.main._FIRST_LOOP", False))
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_trading_config",
                    return_value=SimpleNamespace(
                        debug=False,
                        debug_price_to_beat=False,
                        use_recommended_limit=False,
                        max_trades_per_period=1,
                        max_losses_per_run=0,
                        minimum_wallet_balance=0.0,
                        paper_trading=False,
                    ),
                )
            )
            stack.enter_context(
                patch("custom.btc_agent.main.find_current_btc_updown_market", return_value=market)
            )
            stack.enter_context(patch("custom.btc_agent.main.sync_period_state", return_value=False))
            stack.enter_context(patch("custom.btc_agent.main.get_state", return_value=state))
            stack.enter_context(
                patch("custom.btc_agent.main.resolve_price_to_beat_with_retries", return_value=market)
            )
            stack.enter_context(
                patch(
                    "custom.btc_agent.main.get_account_balance_snapshot",
                    return_value=SimpleNamespace(cash_balance=100.0, total_account_value=100.0),
                )
            )
            stack.enter_context(
                patch("custom.btc_agent.main.get_feature_readiness", return_value=(True, None))
            )
            stack.enter_context(
                patch("custom.btc_agent.main.build_btc_features", return_value=features)
            )
            stack.enter_context(
                patch("custom.btc_agent.main.get_active_orders", return_value=[])
            )
            stack.enter_context(
                patch("custom.btc_agent.main.get_token_quote_snapshot", return_value=up_snapshot)
            )
            stack.enter_context(patch("custom.btc_agent.main.print_market_context"))
            stack.enter_context(patch("custom.btc_agent.main.print_features"))
            stack.enter_context(patch("custom.btc_agent.main.print_active_orders"))
            stack.enter_context(patch("custom.btc_agent.main.decide_trade", return_value=decision))
            stack.enter_context(patch("custom.btc_agent.main.maybe_execute_trade", return_value=result))
            mock_record = stack.enter_context(patch("custom.btc_agent.main.record_executed_trade"))
            stack.enter_context(patch("custom.btc_agent.main.os.getcwd", return_value="/appl/agents"))

            run_once()

        mock_record.assert_called_once()
        tracked_order = mock_record.call_args.args[0]
        self.assertEqual(tracked_order.live_order_id, "order-123")
        self.assertEqual(tracked_order.api_order_state, "ORDER_STATE_NEW")
        self.assertIsNone(tracked_order.actual_fill_price)
        self.assertFalse(
            os.path.exists(
                "/appl/agents/completed_orders/completed_order_win_up_1778208300.txt"
            )
        )
        self.assertFalse(
            os.path.exists(
                "/appl/agents/completed_orders/completed_order_unfilled_up_1778208300.txt"
            )
        )

    def test_run_once_finalizes_previous_orders_with_prior_slug_resolution_price(self):
        previous_order = ActivePaperOrder(
            market_slug="btc-updown-5m-1777513500",
            market_title="Prior Period",
            side="DOWN",
            shares=5.0,
            entry_price=0.43,
            token_id="down-token",
            target_btc_price=77720.0,
            entry_btc_price=77725.0,
            actual_fill_price=0.43,
            filled=True,
        )
        market = SimpleNamespace(
            slug="btc-updown-5m-1777513800",
            title="Bitcoin Up or Down",
            settlement_threshold=None,
            up_token_id="up-token",
            down_token_id="down-token",
        )
        previous_state = SimpleNamespace(
            market_slug="btc-updown-5m-1777513500",
            active_orders=[previous_order],
        )
        current_state = SimpleNamespace(trades_executed=1, active_orders=[])

        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(
                debug=False,
                debug_price_to_beat=False,
                max_trades_per_period=1,
                max_losses_per_run=0,
                minimum_wallet_balance=0.0,
            ),
        ), patch(
            "custom.btc_agent.main.find_current_btc_updown_market",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.get_state",
            side_effect=[previous_state, current_state],
        ), patch(
            "custom.btc_agent.main.sync_period_state",
            return_value=True,
        ), patch(
            "custom.btc_agent.main.resolve_price_to_beat_with_retries",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.get_account_balance_snapshot",
            return_value=SimpleNamespace(cash_balance=100.0),
        ), patch(
            "custom.btc_agent.main.print_account_snapshot_from_snapshot",
        ), patch(
            "custom.btc_agent.main.enforce_minimum_wallet_balance",
        ), patch(
            "custom.btc_agent.main.fetch_btc_resolution_price_for_slug",
            return_value=77710.0,
        ) as mock_resolution_price, patch(
            "custom.btc_agent.main.fetch_btc_spot_price",
            return_value=77799.0,
        ), patch(
            "custom.btc_agent.main.print_active_orders",
        ), patch(
            "custom.btc_agent.main.clear_price_to_beat_debug_files",
        ), patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/appl/agents",
        ), patch(
            "custom.btc_agent.main.sys.exit",
            side_effect=SystemExit(1),
        ):
            with self.assertRaises(SystemExit):
                run_once()

        mock_resolution_price.assert_called_once_with("btc-updown-5m-1777513500")
        with open(
            "/appl/agents/completed_orders/completed_order_win_down_1777513500.txt",
            encoding="utf-8",
        ) as order_file:
            content = order_file.read()

        self.assertIn("current_btc_price=77710.00", content)

    def test_run_once_finalizes_previous_orders_with_new_slug_price_to_beat_first(self):
        previous_order = ActivePaperOrder(
            market_slug="btc-updown-5m-1777513500",
            market_title="Prior Period",
            side="DOWN",
            shares=5.0,
            entry_price=0.43,
            token_id="down-token",
            target_btc_price=77720.0,
            entry_btc_price=77725.0,
            actual_fill_price=0.43,
            filled=True,
        )
        market = SimpleNamespace(
            slug="btc-updown-5m-1777513800",
            title="Bitcoin Up or Down",
            settlement_threshold=77710.0,
            up_token_id="up-token",
            down_token_id="down-token",
        )
        previous_state = SimpleNamespace(
            market_slug="btc-updown-5m-1777513500",
            active_orders=[previous_order],
        )
        current_state = SimpleNamespace(trades_executed=1, active_orders=[])

        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(
                debug=False,
                debug_price_to_beat=False,
                max_trades_per_period=1,
                max_losses_per_run=0,
                minimum_wallet_balance=0.0,
            ),
        ), patch(
            "custom.btc_agent.main.find_current_btc_updown_market",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.get_state",
            side_effect=[previous_state, current_state],
        ), patch(
            "custom.btc_agent.main.sync_period_state",
            return_value=True,
        ), patch(
            "custom.btc_agent.main.resolve_price_to_beat_with_retries",
            return_value=market,
        ), patch(
            "custom.btc_agent.main.get_account_balance_snapshot",
            return_value=SimpleNamespace(cash_balance=100.0),
        ), patch(
            "custom.btc_agent.main.print_account_snapshot_from_snapshot",
        ), patch(
            "custom.btc_agent.main.enforce_minimum_wallet_balance",
        ), patch(
            "custom.btc_agent.main.fetch_btc_resolution_price_for_slug",
            return_value=77799.0,
        ) as mock_resolution_price, patch(
            "custom.btc_agent.main.fetch_btc_spot_price",
            return_value=77888.0,
        ), patch(
            "custom.btc_agent.main.print_active_orders",
        ), patch(
            "custom.btc_agent.main.clear_price_to_beat_debug_files",
        ), patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/appl/agents",
        ):
            run_once()

        mock_resolution_price.assert_not_called()
        with open(
            "/appl/agents/completed_orders/completed_order_win_down_1777513500.txt",
            encoding="utf-8",
        ) as order_file:
            content = order_file.read()

        self.assertIn("current_btc_price=77710.00", content)

    def test_enforce_session_loss_trade_limit_does_not_exit_below_cap_even_with_active_orders(self):
        from custom.btc_agent import main as main_module
        main_module._SESSION_LOSS_TRADES = 2

        with patch(
            "custom.btc_agent.main.sys.exit",
        ) as mock_exit:
            enforce_session_loss_trade_limit(
                SimpleNamespace(max_losses_per_run=3)
            )

        mock_exit.assert_not_called()

    def test_main_llm_connection_debug_bypasses_geolocation_and_exits_successfully(self):
        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(
                debug=False,
                paper_trading=True,
                llm_connection_debug=True,
            ),
        ), patch(
            "custom.btc_agent.main.describe_proxy_configuration",
            return_value="disabled via USE_PROXY=false",
        ), patch(
            "custom.btc_agent.main.test_llm_connection",
            return_value=(True, "LLM connection test succeeded (openai/gpt-4.1-mini)"),
        ) as mock_test_llm_connection, patch(
            "custom.btc_agent.main.enforce_allowed_ip_location",
        ) as mock_enforce_allowed_ip_location, patch(
            "builtins.print",
        ) as mock_print:
            main()

        printed_lines = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
        self.assertTrue(any("LLM connection debug mode enabled." in line for line in printed_lines))
        self.assertTrue(any("LLM connection test: LLM connection test succeeded" in line for line in printed_lines))
        mock_test_llm_connection.assert_called_once()
        mock_enforce_allowed_ip_location.assert_not_called()

    def test_main_llm_connection_debug_exits_nonzero_on_failure(self):
        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(
                debug=False,
                paper_trading=True,
                llm_connection_debug=True,
            ),
        ), patch(
            "custom.btc_agent.main.describe_proxy_configuration",
            return_value="disabled via USE_PROXY=false",
        ), patch(
            "custom.btc_agent.main.test_llm_connection",
            return_value=(False, "Gemini request failed: offline"),
        ), patch(
            "custom.btc_agent.main.enforce_allowed_ip_location",
        ) as mock_enforce_allowed_ip_location, patch(
            "builtins.print",
        ):
            with self.assertRaises(SystemExit) as exc:
                main()

        self.assertEqual(exc.exception.code, 1)
        mock_enforce_allowed_ip_location.assert_not_called()

    def test_main_exits_cleanly_when_quit_requested_during_sleep(self):
        fake_monitor = SimpleNamespace(poll_quit_requested=lambda: True)

        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(
                debug=False,
                paper_trading=True,
                llm_connection_debug=False,
            ),
        ), patch(
            "custom.btc_agent.main.describe_proxy_configuration",
            return_value="disabled via USE_PROXY=false",
        ), patch(
            "custom.btc_agent.main.enforce_allowed_ip_location",
        ), patch(
            "custom.btc_agent.main.get_account_balance_snapshot",
            return_value=SimpleNamespace(cash_balance=10.0),
        ), patch(
            "custom.btc_agent.main.enforce_minimum_wallet_balance",
        ), patch(
            "custom.btc_agent.main.run_once",
        ) as mock_run_once, patch(
            "custom.btc_agent.main.QuitKeyMonitor",
        ) as mock_quit_key_monitor, patch(
            "builtins.print",
        ) as mock_print:
            mock_quit_key_monitor.return_value.__enter__.return_value = fake_monitor
            mock_quit_key_monitor.return_value.__exit__.return_value = None
            main()

        printed_lines = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
        self.assertTrue(any("Press q to quit." in line for line in printed_lines))
        self.assertTrue(any("Quit requested via keyboard. Exiting BTC agent." in line for line in printed_lines))
        mock_run_once.assert_not_called()


if __name__ == "__main__":
    unittest.main()
