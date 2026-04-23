import sys
import types
import unittest

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))
sys.modules.setdefault(
    "agents.polymarket.polymarket",
    types.SimpleNamespace(Polymarket=object),
)

from custom.btc_agent.main import has_valid_price_to_beat


class TestBtcMain(unittest.TestCase):
    def test_has_valid_price_to_beat_rejects_none_and_small_values(self):
        self.assertFalse(has_valid_price_to_beat(None))
        self.assertFalse(has_valid_price_to_beat(1))
        self.assertFalse(has_valid_price_to_beat(5))

    def test_has_valid_price_to_beat_accepts_realistic_btc_values(self):
        self.assertTrue(has_valid_price_to_beat(78218.01972274295))


if __name__ == "__main__":
    unittest.main()
