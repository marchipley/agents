import unittest

from custom.btc_agent.timing import is_last_minute_of_market, seconds_remaining_in_market


class TestBtcTiming(unittest.TestCase):
    def test_seconds_remaining_in_market_never_negative(self):
        self.assertEqual(seconds_remaining_in_market(100, now_ts=150), 0)

    def test_is_last_minute_of_market_true_at_sixty_seconds_or_less(self):
        self.assertTrue(is_last_minute_of_market(160, now_ts=100))
        self.assertTrue(is_last_minute_of_market(159, now_ts=100))

    def test_is_last_minute_of_market_false_above_sixty_seconds(self):
        self.assertFalse(is_last_minute_of_market(161, now_ts=100))


if __name__ == "__main__":
    unittest.main()
