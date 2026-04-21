import unittest
from unittest.mock import patch

import requests

from custom.btc_agent import indicators


class TestBtcIndicators(unittest.TestCase):
    def setUp(self):
        indicators._PRICE_HISTORY.clear()
        indicators._LAST_SUCCESSFUL_PROVIDER_INDEX = 0

    def test_fetch_btc_spot_price_uses_secondary_provider_after_primary_failure(self):
        with patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_coingecko",
            side_effect=requests.HTTPError("429 Too Many Requests"),
        ), patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_coinbase",
            return_value=75123.0,
        ):
            price = indicators.fetch_btc_spot_price()

        self.assertEqual(price, 75123.0)
        self.assertEqual(indicators.get_latest_cached_price(), 75123.0)
        self.assertEqual(indicators._LAST_SUCCESSFUL_PROVIDER_INDEX, 1)

    def test_fetch_btc_spot_price_uses_cached_value_when_all_providers_fail(self):
        indicators._record_price_sample(75000.0)

        with patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_coingecko",
            side_effect=requests.HTTPError("429 Too Many Requests"),
        ), patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_coinbase",
            side_effect=requests.HTTPError("503 Service Unavailable"),
        ):
            price = indicators.fetch_btc_spot_price()

        self.assertEqual(price, 75000.0)

    def test_fetch_btc_spot_price_raises_without_cache_when_all_providers_fail(self):
        with patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_coingecko",
            side_effect=requests.HTTPError("429 Too Many Requests"),
        ), patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_coinbase",
            side_effect=requests.HTTPError("429 Too Many Requests"),
        ):
            with self.assertRaises(requests.HTTPError):
                indicators.fetch_btc_spot_price()


if __name__ == "__main__":
    unittest.main()
