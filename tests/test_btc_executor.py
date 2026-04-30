import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))
sys.modules.setdefault(
    "agents.polymarket.polymarket",
    types.SimpleNamespace(Polymarket=object),
)

from custom.btc_agent.executor import (
    _get_order_notional,
    _scale_live_size_for_min_notional,
    evaluate_ok_to_submit,
    get_account_balance_snapshot,
    get_submission_limit_price,
    get_submission_limit_label,
)


class TestBtcExecutor(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
