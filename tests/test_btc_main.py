import sys
import types
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))
sys.modules.setdefault(
    "agents.polymarket.polymarket",
    types.SimpleNamespace(Polymarket=object),
)

from custom.btc_agent.main import (
    clear_price_to_beat_debug_files,
    has_valid_price_to_beat,
    run_once,
    main,
    resolve_price_to_beat_with_retries,
    wait_for_next_tick_or_quit,
    write_price_to_beat_debug_file,
)


class TestBtcMain(unittest.TestCase):
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

    def test_write_price_to_beat_debug_file_writes_report(self):
        with patch(
            "custom.btc_agent.main.build_price_to_beat_debug_reports",
            return_value=["page debug report\n", "next data debug report\n", "third page debug report\n"],
        ), patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/appl/agents",
        ), patch(
            "custom.btc_agent.main._DEBUG_WRITTEN_SLUGS",
            set(),
        ):
            write_price_to_beat_debug_file("btc-updown-5m-1776983100")

        with open("/appl/agents/logs/priceToBeatDebug.txt", encoding="utf-8") as debug_file:
            self.assertEqual(debug_file.read(), "page debug report\n")
        with open("/appl/agents/logs/priceToBeatDebugPg2.txt", encoding="utf-8") as debug_file:
            self.assertEqual(debug_file.read(), "next data debug report\n")
        with open("/appl/agents/logs/priceToBeatDebugPg3.txt", encoding="utf-8") as debug_file:
            self.assertEqual(debug_file.read(), "third page debug report\n")

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

        mock_build_reports.assert_called_once_with("btc-updown-5m-1776983100")

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
        ), patch(
            "custom.btc_agent.main.os.remove",
        ) as mock_remove:
            clear_price_to_beat_debug_files()

        removed_paths = [call.args[0] for call in mock_remove.call_args_list]
        self.assertIn("/appl/agents/logs/priceToBeatDebug.txt", removed_paths)
        self.assertIn("/appl/agents/logs/priceToBeatDebugPg2.txt", removed_paths)
        self.assertNotIn("/appl/agents/logs/unrelated.log", removed_paths)

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

    def test_resolve_price_to_beat_with_retries_skips_retries_when_debug_price_to_beat_enabled(self):
        initial_market = SimpleNamespace(slug="btc-updown-5m-1777056000", settlement_threshold=None)

        with patch(
            "custom.btc_agent.main.get_trading_config",
            return_value=SimpleNamespace(debug_price_to_beat=True),
        ), patch(
            "custom.btc_agent.main.get_btc_updown_market_by_slug",
        ) as mock_get_by_slug, patch(
            "custom.btc_agent.main.time.sleep",
        ):
            market = resolve_price_to_beat_with_retries(initial_market, retry_attempts=2, retry_delay_seconds=1)

        self.assertIsNone(market.settlement_threshold)
        mock_get_by_slug.assert_not_called()

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

        mock_write_debug.assert_called_once_with("btc-updown-5m-1777056000")

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

        mock_clear_debug.assert_called_once_with()

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
