import unittest
from datetime import datetime, timezone
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

    def test_build_btc_features_uses_current_window_samples_for_window_open(self):
        indicators._record_price_sample(
            75781.0,
            as_of=datetime.fromtimestamp(1_776_813_300, tz=timezone.utc),
        )
        indicators._record_price_sample(
            75854.11,
            as_of=datetime.fromtimestamp(1_776_813_600, tz=timezone.utc),
        )

        with patch(
            "custom.btc_agent.indicators.fetch_btc_spot_price",
            return_value=75821.82,
        ):
            features = indicators.build_btc_features(window_start_ts=1_776_813_600)

        self.assertEqual(features.window_open_price, 75854.11)
        self.assertAlmostEqual(
            features.delta_pct_from_window_open,
            (75821.82 - 75854.11) / 75854.11,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
