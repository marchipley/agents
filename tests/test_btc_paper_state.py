import unittest

from custom.btc_agent.paper_state import (
    ActivePaperOrder,
    classify_position,
    describe_target,
    get_state,
    record_executed_trade,
    sync_period_state,
)


class TestBtcPaperState(unittest.TestCase):
    def setUp(self):
        sync_period_state("btc-updown-5m-1", "Period 1")

    def test_record_trade_updates_period_state(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1",
            market_title="Period 1",
            side="UP",
            shares=5.0,
            entry_price=0.55,
            token_id="token-1",
            target_btc_price=75000.0,
            entry_btc_price=75010.0,
        )

        record_executed_trade(order)
        state = get_state()

        self.assertEqual(state.trades_executed, 1)
        self.assertEqual(len(state.active_orders), 1)
        self.assertEqual(state.active_orders[0].token_id, "token-1")

    def test_sync_period_state_resets_trade_count_and_orders(self):
        record_executed_trade(
            ActivePaperOrder(
                market_slug="btc-updown-5m-1",
                market_title="Period 1",
                side="UP",
                shares=5.0,
                entry_price=0.55,
                token_id="token-1",
                target_btc_price=75000.0,
                entry_btc_price=75010.0,
            )
        )

        changed = sync_period_state("btc-updown-5m-2", "Period 2")
        state = get_state()

        self.assertTrue(changed)
        self.assertEqual(state.trades_executed, 0)
        self.assertEqual(state.active_orders, [])

    def test_classify_position_uses_target_direction(self):
        up_order = ActivePaperOrder(
            market_slug="btc-updown-5m-1",
            market_title="Period 1",
            side="UP",
            shares=5.0,
            entry_price=0.55,
            token_id="token-up",
            target_btc_price=75000.0,
            entry_btc_price=75010.0,
        )
        down_order = ActivePaperOrder(
            market_slug="btc-updown-5m-1",
            market_title="Period 1",
            side="DOWN",
            shares=5.0,
            entry_price=0.45,
            token_id="token-down",
            target_btc_price=75000.0,
            entry_btc_price=74990.0,
        )

        self.assertEqual(classify_position(up_order, 75010.0), "WINNING")
        self.assertEqual(classify_position(up_order, 74990.0), "LOSING")
        self.assertEqual(classify_position(down_order, 74990.0), "WINNING")
        self.assertEqual(classify_position(down_order, 75000.0), "TIED")

    def test_describe_target_marks_approximate_threshold(self):
        order = ActivePaperOrder(
            market_slug="btc-updown-5m-1",
            market_title="Period 1",
            side="DOWN",
            shares=5.0,
            entry_price=0.45,
            token_id="token-down",
            target_btc_price=75000.0,
            entry_btc_price=74990.0,
            target_is_approximate=True,
        )

        self.assertEqual(
            describe_target(order),
            "BTC must finish below approximately 75000.00",
        )


if __name__ == "__main__":
    unittest.main()
