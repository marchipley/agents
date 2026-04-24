import os
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import requests

from custom.btc_agent import indicators


class TestBtcIndicators(unittest.TestCase):
    def setUp(self):
        indicators._PRICE_HISTORY.clear()
        indicators._LAST_SUCCESSFUL_PROVIDER_INDEX = 0
        indicators._PRICE_HISTORY_BACKFILLED = False
        indicators._LATEST_PRICE_SOURCE = "unknown"

    @staticmethod
    def _recorded_price_return(price: float):
        def _side_effect(*args, **kwargs):
            indicators._record_price_sample(price)
            return price

        return _side_effect

    def test_fetch_btc_spot_price_uses_secondary_provider_after_primary_failure(self):
        providers = [
            ("Polymarket RTDS", MagicMock(side_effect=requests.HTTPError("socket timeout"))),
            ("Coinbase", MagicMock(return_value=75123.0)),
        ]

        with patch(
            "custom.btc_agent.indicators._get_price_providers",
            return_value=providers,
        ):
            price = indicators.fetch_btc_spot_price()

        self.assertEqual(price, 75123.0)
        self.assertEqual(indicators.get_latest_cached_price(), 75123.0)
        self.assertEqual(indicators._LAST_SUCCESSFUL_PROVIDER_INDEX, 1)
        self.assertEqual(indicators._LATEST_PRICE_SOURCE, "Coinbase")

    def test_fetch_btc_spot_price_uses_cached_value_when_all_providers_fail(self):
        indicators._record_price_sample(
            75000.0,
            as_of=datetime.now(timezone.utc),
        )

        providers = [
            ("Polymarket RTDS", MagicMock(side_effect=requests.HTTPError("socket timeout"))),
            ("CoinGecko", MagicMock(side_effect=requests.HTTPError("429 Too Many Requests"))),
            ("Coinbase", MagicMock(side_effect=requests.HTTPError("503 Service Unavailable"))),
        ]

        with patch(
            "custom.btc_agent.indicators._get_price_providers",
            return_value=providers,
        ):
            price = indicators.fetch_btc_spot_price()

        self.assertEqual(price, 75000.0)
        self.assertTrue(indicators._LATEST_PRICE_SOURCE.startswith("cache:"))

    def test_fetch_btc_spot_price_retries_rtds_first_on_each_call(self):
        rtds_provider = MagicMock(return_value=78019.41)
        fallback_provider = MagicMock(return_value=75123.0)
        indicators._LAST_SUCCESSFUL_PROVIDER_INDEX = 1

        with patch(
            "custom.btc_agent.indicators._get_price_providers",
            return_value=[
                ("Polymarket RTDS", rtds_provider),
                ("Coinbase", fallback_provider),
            ],
        ):
            price = indicators.fetch_btc_spot_price()

        self.assertEqual(price, 78019.41)
        rtds_provider.assert_called_once()
        fallback_provider.assert_not_called()
        self.assertEqual(indicators._LAST_SUCCESSFUL_PROVIDER_INDEX, 0)
        self.assertEqual(indicators._LATEST_PRICE_SOURCE, "Polymarket RTDS")

    def test_fetch_btc_spot_price_raises_when_cache_is_stale(self):
        indicators._record_price_sample(
            75000.0,
            as_of=datetime.now(timezone.utc) - timedelta(seconds=30),
        )

        providers = [
            ("Polymarket RTDS", MagicMock(side_effect=requests.HTTPError("socket timeout"))),
            ("CoinGecko", MagicMock(side_effect=requests.HTTPError("429 Too Many Requests"))),
            ("Coinbase", MagicMock(side_effect=requests.HTTPError("503 Service Unavailable"))),
        ]

        with patch(
            "custom.btc_agent.indicators._get_price_providers",
            return_value=providers,
        ):
            with self.assertRaises(requests.HTTPError):
                indicators.fetch_btc_spot_price()

    def test_fetch_btc_spot_price_raises_without_cache_when_all_providers_fail(self):
        providers = [
            ("Polymarket RTDS", MagicMock(side_effect=requests.HTTPError("socket timeout"))),
            ("CoinGecko", MagicMock(side_effect=requests.HTTPError("429 Too Many Requests"))),
            ("Coinbase", MagicMock(side_effect=requests.HTTPError("429 Too Many Requests"))),
        ]

        with patch(
            "custom.btc_agent.indicators._get_price_providers",
            return_value=providers,
        ):
            with self.assertRaises(requests.HTTPError):
                indicators.fetch_btc_spot_price()

    def test_fetch_spot_price_from_polymarket_rtds_uses_matching_btc_update(self):
        fake_socket = MagicMock()
        fake_socket.recv.side_effect = [
            json.dumps(
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "payload": {"symbol": "ethusdt", "value": 3200.0},
                }
            ),
            json.dumps(
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "payload": {"symbol": "btcusdt", "value": 78019.41},
                }
            ),
        ]

        with patch(
            "custom.btc_agent.indicators._create_polymarket_rtds_connection",
            return_value=fake_socket,
        ):
            price = indicators._fetch_spot_price_from_polymarket_rtds()

        self.assertEqual(price, 78019.41)
        subscribe_payload = json.loads(fake_socket.send.call_args.args[0])
        self.assertEqual(subscribe_payload["action"], "subscribe")
        self.assertEqual(subscribe_payload["subscriptions"][0]["topic"], "crypto_prices")
        self.assertEqual(
            json.loads(subscribe_payload["subscriptions"][0]["filters"]),
            {"symbol": "btcusdt"},
        )
        fake_socket.close.assert_called_once()

    def test_fetch_spot_price_from_polymarket_rtds_uses_subscribe_snapshot(self):
        fake_socket = MagicMock()
        fake_socket.recv.return_value = json.dumps(
            {
                "topic": "crypto_prices",
                "type": "subscribe",
                "payload": {
                    "symbol": "btcusdt",
                    "data": [
                        {"timestamp": 1, "value": 77715.34},
                        {"timestamp": 2, "value": 77753.41},
                    ],
                },
            }
        )

        with patch(
            "custom.btc_agent.indicators._create_polymarket_rtds_connection",
            return_value=fake_socket,
        ):
            price = indicators._fetch_spot_price_from_polymarket_rtds()

        self.assertEqual(price, 77753.41)
        fake_socket.close.assert_called_once()

    def test_create_polymarket_rtds_connection_clears_proxy_env(self):
        fake_socket = MagicMock()
        observed_direct_all_proxy = []

        def _create_connection(*args, **kwargs):
            observed_direct_all_proxy.append(os.environ.get("ALL_PROXY"))
            return fake_socket

        fake_websocket_module = MagicMock()
        fake_websocket_module.create_connection.side_effect = _create_connection

        with patch.dict(os.environ, {"ALL_PROXY": "socks5h://10.64.0.1:1080"}, clear=False):
            with patch(
                "custom.btc_agent.indicators.websocket",
                fake_websocket_module,
            ):
                connection = indicators._create_polymarket_rtds_connection()
                self.assertEqual(os.environ.get("ALL_PROXY"), "socks5h://10.64.0.1:1080")

        self.assertIs(connection, fake_socket)
        self.assertEqual(observed_direct_all_proxy, [None])

    def test_build_btc_features_uses_current_window_samples_for_window_open(self):
        indicators._record_price_sample(
            75781.0,
            as_of=datetime.fromtimestamp(1_776_813_300, tz=timezone.utc),
        )
        indicators._record_price_sample(
            75854.11,
            as_of=datetime.fromtimestamp(1_776_813_600, tz=timezone.utc),
        )
        indicators._record_price_sample(
            75810.50,
            as_of=datetime.fromtimestamp(1_776_813_650, tz=timezone.utc),
        )

        with patch(
            "custom.btc_agent.indicators.fetch_btc_spot_price",
            side_effect=self._recorded_price_return(75821.82),
        ), patch(
            "custom.btc_agent.indicators.datetime",
        ) as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(1_776_813_660, tz=timezone.utc)
            mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
            mock_datetime.timezone = timezone
            features = indicators.build_btc_features(window_start_ts=1_776_813_600)

        self.assertEqual(features.window_open_price, 75854.11)
        self.assertEqual(features.trailing_5m_open_price, 75854.11)
        self.assertEqual(features.price_source, "unknown")
        self.assertAlmostEqual(
            features.delta_pct_from_window_open,
            (75821.82 - 75854.11) / 75854.11,
            places=6,
        )
        self.assertAlmostEqual(
            features.delta_pct_from_trailing_5m_open,
            (75821.82 - 75854.11) / 75854.11,
            places=6,
        )
        self.assertAlmostEqual(features.delta_from_previous_tick, 11.32, places=2)
        self.assertAlmostEqual(features.momentum_1m, -32.29, places=2)
        self.assertAlmostEqual(features.momentum_5m, -32.29, places=2)
        self.assertEqual(features.retained_sample_count, 4)
        self.assertEqual(features.window_sample_count, 3)
        self.assertEqual(features.trailing_5m_sample_count, 3)

    def test_build_btc_features_carries_forward_last_pre_window_sample(self):
        indicators._record_price_sample(
            77947.30,
            as_of=datetime.fromtimestamp(1_776_968_690, tz=timezone.utc),
        )

        with patch(
            "custom.btc_agent.indicators.fetch_btc_spot_price",
            side_effect=self._recorded_price_return(77948.77),
        ), patch(
            "custom.btc_agent.indicators.datetime",
        ) as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(1_776_968_708, tz=timezone.utc)
            mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
            mock_datetime.timezone = timezone
            features = indicators.build_btc_features(window_start_ts=1_776_968_700)

        self.assertEqual(features.window_open_price, 77947.30)
        self.assertEqual(features.trailing_5m_open_price, 77947.30)
        self.assertAlmostEqual(features.delta_from_previous_tick, 1.47, places=2)
        self.assertAlmostEqual(features.momentum_5m, 1.47, places=2)
        self.assertEqual(features.window_sample_count, 2)
        self.assertEqual(features.trailing_5m_sample_count, 2)

    def test_estimate_market_window_reference_price_prefers_boundary_sample_before_start(self):
        indicators._record_price_sample(
            77940.12,
            as_of=datetime.fromtimestamp(1_776_968_690, tz=timezone.utc),
        )
        indicators._record_price_sample(
            77955.55,
            as_of=datetime.fromtimestamp(1_776_968_708, tz=timezone.utc),
        )

        reference_price = indicators.estimate_market_window_reference_price(
            1_776_968_700,
            now=datetime.fromtimestamp(1_776_968_708, tz=timezone.utc),
        )

        self.assertEqual(reference_price, 77940.12)

    def test_estimate_market_window_reference_price_falls_forward_when_no_prior_sample_exists(self):
        indicators._record_price_sample(
            77955.55,
            as_of=datetime.fromtimestamp(1_776_968_708, tz=timezone.utc),
        )

        reference_price = indicators.estimate_market_window_reference_price(
            1_776_968_700,
            now=datetime.fromtimestamp(1_776_968_708, tz=timezone.utc),
        )

        self.assertEqual(reference_price, 77955.55)

    def test_feature_readiness_false_until_rsi_and_window_warmup_ready(self):
        features = indicators.BtcFeatures(
            as_of=datetime.now(timezone.utc),
            price_usd=75000.0,
            price_source="Coinbase",
            window_open_price=74990.0,
            trailing_5m_open_price=74980.0,
            delta_pct_from_window_open=0.0,
            delta_pct_from_trailing_5m_open=0.0,
            delta_from_previous_tick=None,
            rsi_14=None,
            momentum_1m=None,
            momentum_5m=None,
            volatility_5m=None,
            retained_sample_count=3,
            window_sample_count=1,
            trailing_5m_sample_count=1,
        )

        ready, reason = indicators.get_feature_readiness(features)

        self.assertFalse(ready)
        self.assertIn("RSI warmup incomplete", reason)
        self.assertIn("trailing 5-minute warmup incomplete", reason)

    def test_feature_readiness_true_when_all_features_available(self):
        features = indicators.BtcFeatures(
            as_of=datetime.now(timezone.utc),
            price_usd=75000.0,
            price_source="Coinbase",
            window_open_price=74990.0,
            trailing_5m_open_price=74980.0,
            delta_pct_from_window_open=0.0,
            delta_pct_from_trailing_5m_open=0.0,
            delta_from_previous_tick=3.0,
            rsi_14=55.0,
            momentum_1m=8.0,
            momentum_5m=10.0,
            volatility_5m=6.0,
            retained_sample_count=20,
            window_sample_count=3,
            trailing_5m_sample_count=3,
        )

        ready, reason = indicators.get_feature_readiness(features)

        self.assertTrue(ready)
        self.assertEqual(reason, "ready")

    def test_ensure_price_history_backfilled_seeds_from_recent_trades(self):
        now = datetime.fromtimestamp(1_776_813_660, tz=timezone.utc)
        trades = [
            (
                datetime.fromtimestamp(1_776_813_360 + (idx * 20), tz=timezone.utc),
                75000.0 + idx,
            )
            for idx in range(15)
        ]

        with patch(
            "custom.btc_agent.indicators._fetch_recent_trades_from_coinbase",
            return_value=trades,
        ):
            indicators.ensure_price_history_backfilled(now)

        self.assertTrue(indicators._PRICE_HISTORY_BACKFILLED)
        self.assertGreaterEqual(len(indicators._PRICE_HISTORY), 15)

    def test_ensure_price_history_backfilled_falls_back_to_candles(self):
        now = datetime.fromtimestamp(1_776_813_660, tz=timezone.utc)
        candles = [
            (
                datetime.fromtimestamp(1_776_813_360 + (idx * 60), tz=timezone.utc),
                75000.0 + idx,
                75010.0 + idx,
            )
            for idx in range(5)
        ]

        with patch(
            "custom.btc_agent.indicators._fetch_recent_trades_from_coinbase",
            side_effect=requests.RequestException("trades unavailable"),
        ), patch(
            "custom.btc_agent.indicators._fetch_coinbase_candles",
            return_value=candles,
        ):
            indicators.ensure_price_history_backfilled(now)

        self.assertTrue(indicators._PRICE_HISTORY_BACKFILLED)
        self.assertGreaterEqual(len(indicators._PRICE_HISTORY), 15)


if __name__ == "__main__":
    unittest.main()
