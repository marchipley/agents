import json
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import requests

from custom.btc_agent import indicators


class TestBtcIndicators(unittest.TestCase):
    def setUp(self):
        indicators._PRICE_HISTORY.clear()
        indicators._LAST_SUCCESSFUL_PROVIDER_INDEX = 0
        indicators._PRICE_HISTORY_BACKFILLED = False

    @staticmethod
    def _recorded_price_return(price: float):
        def _side_effect(*args, **kwargs):
            indicators._record_price_sample(price)
            return price

        return _side_effect

    def test_fetch_btc_spot_price_uses_secondary_provider_after_primary_failure(self):
        with patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_polymarket_rtds",
            side_effect=requests.HTTPError("429 Too Many Requests"),
        ), patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_binance_websocket",
            return_value=75123.0,
        ), patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_coinbase",
            return_value=75200.0,
        ):
            price = indicators.fetch_btc_spot_price()

        self.assertEqual(price, 75123.0)
        self.assertEqual(indicators.get_latest_cached_price(), 75123.0)
        self.assertEqual(indicators._LAST_SUCCESSFUL_PROVIDER_INDEX, 1)

    def test_fetch_btc_spot_price_uses_cached_value_when_all_providers_fail(self):
        indicators._record_price_sample(75000.0)

        with patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_polymarket_rtds",
            side_effect=requests.HTTPError("429 Too Many Requests"),
        ), patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_binance_websocket",
            side_effect=requests.HTTPError("503 Service Unavailable"),
        ), patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_coinbase",
            side_effect=requests.HTTPError("503 Service Unavailable"),
        ), patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_coingecko",
            side_effect=requests.HTTPError("503 Service Unavailable"),
        ):
            price = indicators.fetch_btc_spot_price()

        self.assertEqual(price, 75000.0)

    def test_fetch_btc_spot_price_raises_without_cache_when_all_providers_fail(self):
        with patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_polymarket_rtds",
            side_effect=requests.HTTPError("429 Too Many Requests"),
        ), patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_binance_websocket",
            side_effect=requests.HTTPError("429 Too Many Requests"),
        ), patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_coinbase",
            side_effect=requests.HTTPError("429 Too Many Requests"),
        ), patch(
            "custom.btc_agent.indicators._fetch_spot_price_from_coingecko",
            side_effect=requests.HTTPError("429 Too Many Requests"),
        ):
            with self.assertRaises(requests.HTTPError):
                indicators.fetch_btc_spot_price()

    def test_fetch_spot_price_from_binance_websocket_parses_ticker_message(self):
        fake_socket = MagicMock()
        fake_socket.recv.return_value = json.dumps(
            {
                "e": "24hrTicker",
                "E": 1_777_000_001_000,
                "s": "BTCUSDT",
                "c": "75927.01",
            }
        )

        with patch(
            "custom.btc_agent.indicators._create_binance_connection",
            return_value=fake_socket,
        ):
            price = indicators._fetch_spot_price_from_binance_websocket()

        self.assertEqual(price, 75927.01)

    def test_fetch_spot_price_from_polymarket_rtds_prefers_live_update_over_snapshot(self):
        fake_socket = MagicMock()
        fake_socket.recv.side_effect = [
            json.dumps(
                {
                    "topic": "crypto_prices",
                    "type": "subscribe",
                    "payload": {
                        "symbol": "btcusdt",
                        "data": [{"timestamp": 1_777_000_000_000, "value": 75920.0}],
                    },
                }
            ),
            json.dumps(
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "payload": {"symbol": "btcusdt", "timestamp": 1_777_000_001_000, "value": 75927.0},
                }
            ),
        ]

        with patch(
            "custom.btc_agent.indicators._create_polymarket_rtds_connection",
            return_value=fake_socket,
        ):
            price = indicators._fetch_spot_price_from_polymarket_rtds()

        self.assertEqual(price, 75927.0)

    def test_fetch_spot_price_from_polymarket_rtds_rejects_stale_snapshot(self):
        fake_socket = MagicMock()
        fake_socket.recv.side_effect = [
            json.dumps(
                {
                    "topic": "crypto_prices",
                    "type": "subscribe",
                    "payload": {
                        "symbol": "btcusdt",
                        "data": [{"timestamp": 1_700_000_000_000, "value": 75920.0}],
                    },
                }
            ),
            TimeoutError("no fresh update"),
        ]

        with patch(
            "custom.btc_agent.indicators._create_polymarket_rtds_connection",
            return_value=fake_socket,
        ), patch(
            "custom.btc_agent.indicators.datetime",
        ) as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(1_777_000_100, tz=timezone.utc)
            mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
            mock_datetime.timezone = timezone
            with self.assertRaises(requests.RequestException):
                indicators._fetch_spot_price_from_polymarket_rtds()

    def test_record_price_sample_skips_near_duplicate_same_price_samples(self):
        sample_time = datetime.fromtimestamp(1_777_000_000, tz=timezone.utc)
        indicators._record_price_sample(75920.0, as_of=sample_time)
        indicators._record_price_sample(75920.0, as_of=sample_time)
        indicators._record_price_sample(75920.0, as_of=sample_time.replace(microsecond=500000))

        self.assertEqual(len(indicators._PRICE_HISTORY), 1)

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

    def test_build_btc_features_includes_micro_velocity_and_flat_tick_count(self):
        seeded_samples = [
            (datetime.fromtimestamp(1_776_968_675, tz=timezone.utc), 77940.0),
            (datetime.fromtimestamp(1_776_968_690, tz=timezone.utc), 77945.0),
            (datetime.fromtimestamp(1_776_968_700, tz=timezone.utc), 77948.0),
            (datetime.fromtimestamp(1_776_968_705, tz=timezone.utc), 77948.0),
            (datetime.fromtimestamp(1_776_968_708, tz=timezone.utc), 77948.0),
        ]
        for sample_time, sample_price in seeded_samples:
            indicators._record_price_sample(sample_price, as_of=sample_time)

        with patch(
            "custom.btc_agent.indicators.fetch_btc_spot_price",
            side_effect=self._recorded_price_return(77950.0),
        ), patch(
            "custom.btc_agent.indicators.datetime",
        ) as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(1_776_968_720, tz=timezone.utc)
            mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
            mock_datetime.timezone = timezone
            features = indicators.build_btc_features(window_start_ts=1_776_968_700)

        self.assertEqual(features.velocity_15s, 2.0)
        self.assertEqual(features.velocity_30s, 5.0)
        self.assertEqual(features.momentum_acceleration, -3.0)
        self.assertEqual(features.consecutive_flat_ticks, 0)
        self.assertEqual(features.consecutive_directional_ticks, 3)

    def test_build_btc_features_directional_streak_ignores_flat_ticks(self):
        seeded_samples = [
            (datetime.fromtimestamp(1_776_968_675, tz=timezone.utc), 77940.0),
            (datetime.fromtimestamp(1_776_968_690, tz=timezone.utc), 77942.0),
            (datetime.fromtimestamp(1_776_968_700, tz=timezone.utc), 77944.0),
            (datetime.fromtimestamp(1_776_968_705, tz=timezone.utc), 77944.0),
            (datetime.fromtimestamp(1_776_968_708, tz=timezone.utc), 77944.0),
        ]
        for sample_time, sample_price in seeded_samples:
            indicators._record_price_sample(sample_price, as_of=sample_time)

        with patch(
            "custom.btc_agent.indicators.fetch_btc_spot_price",
            side_effect=self._recorded_price_return(77946.0),
        ), patch(
            "custom.btc_agent.indicators.datetime",
        ) as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(1_776_968_720, tz=timezone.utc)
            mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
            mock_datetime.timezone = timezone
            features = indicators.build_btc_features(window_start_ts=1_776_968_700)

        self.assertEqual(features.consecutive_flat_ticks, 0)
        self.assertEqual(features.consecutive_directional_ticks, 3)

    def test_build_btc_features_directional_streak_ignores_small_counter_move_noise(self):
        seeded_samples = [
            (datetime.fromtimestamp(1_776_968_675, tz=timezone.utc), 78000.0),
            (datetime.fromtimestamp(1_776_968_690, tz=timezone.utc), 78008.0),
            (datetime.fromtimestamp(1_776_968_700, tz=timezone.utc), 78016.0),
            (datetime.fromtimestamp(1_776_968_705, tz=timezone.utc), 78014.5),
            (datetime.fromtimestamp(1_776_968_708, tz=timezone.utc), 78022.0),
        ]
        for sample_time, sample_price in seeded_samples:
            indicators._record_price_sample(sample_price, as_of=sample_time)

        with patch(
            "custom.btc_agent.indicators.fetch_btc_spot_price",
            side_effect=self._recorded_price_return(78030.0),
        ), patch(
            "custom.btc_agent.indicators.datetime",
        ) as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(1_776_968_720, tz=timezone.utc)
            mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
            mock_datetime.timezone = timezone
            features = indicators.build_btc_features(window_start_ts=1_776_968_700)

        self.assertGreaterEqual(features.consecutive_directional_ticks, 4)

    def test_build_btc_features_populates_phase2_indicator_fields(self):
        base_ts = 1_776_968_300
        for idx in range(24):
            indicators._record_price_sample(
                77900.0 + (idx * 2.0),
                as_of=datetime.fromtimestamp(base_ts + (idx * 20), tz=timezone.utc),
            )

        with patch(
            "custom.btc_agent.indicators.fetch_btc_spot_price",
            side_effect=self._recorded_price_return(77950.0),
        ), patch(
            "custom.btc_agent.indicators.datetime",
        ) as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(base_ts + (24 * 20), tz=timezone.utc)
            mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
            mock_datetime.timezone = timezone
            features = indicators.build_btc_features(window_start_ts=1_776_968_700)

        self.assertIsNotNone(features.rsi_9)
        self.assertIsNotNone(features.rsi_14)
        self.assertIsNotNone(features.rsi_speed_divergence)
        self.assertIsNotNone(features.ema_9)
        self.assertIsNotNone(features.ema_21)
        self.assertEqual(features.ema_alignment, True)
        self.assertEqual(features.ema_cross_direction, "bullish")
        self.assertIsNotNone(features.adx_14)
        self.assertIsNotNone(features.atr_14)

    def test_build_btc_features_rsi_9_and_rsi_14_are_computed_independently(self):
        prices = [
            100.0,
            101.0,
            100.5,
            101.5,
            101.0,
            102.0,
            101.2,
            102.5,
            101.7,
            103.0,
            102.2,
            103.3,
            102.7,
            103.8,
            102.9,
            104.0,
            103.4,
            104.6,
            103.9,
            105.2,
            104.7,
            105.9,
        ]
        base_ts = 1_776_968_300
        for idx, price in enumerate(prices):
            indicators._record_price_sample(
                price,
                as_of=datetime.fromtimestamp(base_ts + (idx * 20), tz=timezone.utc),
            )

        with patch(
            "custom.btc_agent.indicators.fetch_btc_spot_price",
            side_effect=self._recorded_price_return(105.4),
        ), patch(
            "custom.btc_agent.indicators.datetime",
        ) as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(base_ts + (len(prices) * 20), tz=timezone.utc)
            mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
            mock_datetime.timezone = timezone
            features = indicators.build_btc_features(window_start_ts=1_776_968_700)

        self.assertIsNotNone(features.rsi_9)
        self.assertIsNotNone(features.rsi_14)
        self.assertNotEqual(features.rsi_9, features.rsi_14)

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
            window_open_price=74990.0,
            trailing_5m_open_price=74980.0,
            delta_pct_from_window_open=0.0,
            delta_pct_from_trailing_5m_open=0.0,
            delta_from_previous_tick=None,
            rsi_9=None,
            rsi_14=None,
            rsi_speed_divergence=None,
            momentum_1m=None,
            momentum_5m=None,
            velocity_15s=None,
            velocity_30s=None,
            momentum_acceleration=None,
            ema_9=None,
            ema_21=None,
            ema_alignment=None,
            ema_cross_direction=None,
            adx_14=None,
            atr_14=None,
            volatility_5m=None,
            consecutive_flat_ticks=0,
            consecutive_directional_ticks=0,
            retained_sample_count=3,
            window_sample_count=1,
            trailing_5m_sample_count=1,
        )

        ready, reason = indicators.get_feature_readiness(features)

        self.assertFalse(ready)
        self.assertIn("RSI warmup incomplete", reason)
        self.assertIn("trailing 5-minute warmup incomplete", reason)
        self.assertIn("phase 2 indicator warmup incomplete", reason)

    def test_feature_readiness_true_when_all_features_available(self):
        features = indicators.BtcFeatures(
            as_of=datetime.now(timezone.utc),
            price_usd=75000.0,
            window_open_price=74990.0,
            trailing_5m_open_price=74980.0,
            delta_pct_from_window_open=0.0,
            delta_pct_from_trailing_5m_open=0.0,
            delta_from_previous_tick=3.0,
            rsi_9=58.0,
            rsi_14=55.0,
            rsi_speed_divergence=3.0,
            momentum_1m=8.0,
            momentum_5m=10.0,
            velocity_15s=4.0,
            velocity_30s=6.0,
            momentum_acceleration=-2.0,
            ema_9=74995.0,
            ema_21=74980.0,
            ema_alignment=True,
            ema_cross_direction="bullish",
            adx_14=29.0,
            atr_14=11.0,
            volatility_5m=6.0,
            consecutive_flat_ticks=1,
            consecutive_directional_ticks=3,
            retained_sample_count=21,
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
