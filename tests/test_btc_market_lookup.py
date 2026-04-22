import sys
import types
import unittest

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))

from custom.btc_agent.market_lookup import _extract_market_from_event


class TestBtcMarketLookup(unittest.TestCase):
    def test_extract_market_prefers_group_item_threshold(self):
        event = {
            "id": "1",
            "title": "Bitcoin Up or Down - April 22, 1:40PM-1:45PM ET",
            "markets": [
                {
                    "id": "2",
                    "question": "Will Bitcoin finish above or below 78860?",
                    "groupItemThreshold": 78860,
                    "tokens": [
                        {"outcome": "Up", "token_id": "up-token"},
                        {"outcome": "Down", "token_id": "down-token"},
                    ],
                    "start_ts": 1776879600,
                    "end_ts": 1776879900,
                }
            ],
        }

        market = _extract_market_from_event(event, "btc-updown-5m-1776879600")

        self.assertIsNotNone(market)
        self.assertEqual(market.settlement_threshold, 78860.0)
        self.assertEqual(market.question, "Will Bitcoin finish above or below 78860?")

    def test_extract_market_falls_back_to_question_threshold(self):
        event = {
            "id": "1",
            "title": "Bitcoin Up or Down",
            "markets": [
                {
                    "id": "2",
                    "question": "Will Bitcoin finish above 78,860?",
                    "tokens": [
                        {"outcome": "Up", "token_id": "up-token"},
                        {"outcome": "Down", "token_id": "down-token"},
                    ],
                    "start_ts": 1776879600,
                    "end_ts": 1776879900,
                }
            ],
        }

        market = _extract_market_from_event(event, "btc-updown-5m-1776879600")

        self.assertIsNotNone(market)
        self.assertEqual(market.settlement_threshold, 78860.0)


if __name__ == "__main__":
    unittest.main()
