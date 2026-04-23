import sys
import types
import unittest

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))
sys.modules.setdefault(
    "agents.polymarket.polymarket",
    types.SimpleNamespace(Polymarket=object),
)

from custom.btc_agent.executor import _get_order_notional, _scale_live_size_for_min_notional


class TestBtcExecutor(unittest.TestCase):
    def test_scale_live_size_for_min_notional_adds_buffer_above_exchange_minimum(self):
        limit_price = 0.19992

        size = _scale_live_size_for_min_notional(
            base_size=5.0,
            limit_price=limit_price,
            min_order_usd=1.0,
        )
        order_notional = _get_order_notional(size, limit_price)

        self.assertGreaterEqual(order_notional, 1.01)


if __name__ == "__main__":
    unittest.main()
