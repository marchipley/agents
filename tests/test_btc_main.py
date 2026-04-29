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
    has_valid_price_to_beat,
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
            "custom.btc_agent.main.build_price_to_beat_debug_report",
            return_value="debug report\n",
        ), patch(
            "custom.btc_agent.main.os.getcwd",
            return_value="/appl/agents",
        ):
            write_price_to_beat_debug_file("btc-updown-5m-1776983100")

        with open("/appl/agents/logs/priceToBeatDebug.txt", encoding="utf-8") as debug_file:
            self.assertEqual(debug_file.read(), "debug report\n")

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
