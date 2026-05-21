import json
import io
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault("websockets", types.SimpleNamespace(connect=object))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object, create_connection=object))

from custom.btc_agent.market_lookup import (
    BtcUpDownMarket,
    _parse_live_market_probabilities_message,
    _parse_live_market_resolution_price_message,
    _build_current_period_dataset,
    build_price_to_beat_debug_report,
    build_price_to_beat_debug_reports,
    fetch_live_market_probabilities_from_clob_ws,
    fetch_btc_resolution_price_for_slug,
    get_btc_updown_market_by_slug,
    _extract_current_period_open_from_next_data,
    _extract_embedded_next_data_payload,
    _extract_next_build_id,
    _extract_event_from_next_data,
    _extract_live_period_open_from_next_data,
    _extract_market_from_event,
    _extract_previous_period_close_from_next_data,
    _extract_previous_period_final_price_from_next_data,
    _extract_threshold_from_price_to_beat_response,
    _extract_vatic_price_from_response,
    _hydrate_missing_threshold_from_page,
    _fetch_event_from_next_data_route,
    _fetch_next_data_payload_chain,
    _fetch_next_data_payload,
    _fetch_price_to_beat_by_slug,
    _fetch_vatic_price_to_beat_by_slug,
    _fetch_btc_resolution_price_from_clob_ws,
    _fetch_price_via_selenium,
    _write_current_period_dataset_file,
    _extract_threshold_from_page_html,
    _parse_threshold_from_text,
    _fetch_event_by_slug,
)


class TestBtcMarketLookup(unittest.TestCase):
    def test_parse_live_market_resolution_price_message_reads_final_price(self):
        message = json.dumps(
            {
                "event_type": "market_resolved",
                "market": {
                    "slug": "btc-updown-5m-1777513800",
                    "finalPrice": 77763.01,
                },
            }
        )

        self.assertEqual(_parse_live_market_resolution_price_message(message), 77763.01)

    def test_parse_live_market_resolution_price_message_reads_close_price(self):
        message = json.dumps(
            [
                {
                    "event_type": "market_resolved",
                    "closePrice": "77722.39",
                }
            ]
        )

        self.assertEqual(_parse_live_market_resolution_price_message(message), 77722.39)

    def test_fetch_btc_resolution_price_from_clob_ws_subscribes_and_reads_final_price(self):
        fake_ws = Mock()
        fake_ws.recv.return_value = json.dumps({"finalPrice": 77763.01})

        with patch(
            "custom.btc_agent.market_lookup.websocket.create_connection",
            return_value=fake_ws,
        ) as mock_create_connection:
            price = _fetch_btc_resolution_price_from_clob_ws("btc-updown-5m-1777513800")

        self.assertEqual(price, 77763.01)
        mock_create_connection.assert_called_once_with(
            "wss://ws-subscriptions-clob.polymarket.com/ws/market",
            timeout=30.0,
        )
        fake_ws.send.assert_called_once_with(
            json.dumps({"type": "subscribe", "market": "btc-updown-5m-1777513800"})
        )
        fake_ws.close.assert_called_once()

    def test_fetch_btc_resolution_price_for_slug_prefers_websocket_before_rest_fallback(self):
        with patch(
            "custom.btc_agent.market_lookup._fetch_btc_resolution_price_from_clob_ws",
            return_value=77763.01,
        ) as mock_ws, patch(
            "custom.btc_agent.market_lookup._fetch_btc_resolution_price_from_rest_sources",
        ) as mock_rest:
            price = fetch_btc_resolution_price_for_slug("btc-updown-5m-1777513800")

        self.assertEqual(price, 77763.01)
        mock_ws.assert_called_once_with("btc-updown-5m-1777513800")
        mock_rest.assert_not_called()

    def test_fetch_btc_resolution_price_for_slug_uses_rest_when_websocket_has_no_price(self):
        with patch(
            "custom.btc_agent.market_lookup._fetch_btc_resolution_price_from_clob_ws",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_btc_resolution_price_from_rest_sources",
            return_value=77722.39,
        ) as mock_rest:
            price = fetch_btc_resolution_price_for_slug("btc-updown-5m-1777513800")

        self.assertEqual(price, 77722.39)
        mock_rest.assert_called_once_with("btc-updown-5m-1777513800")

    def test_get_btc_updown_market_by_slug_uses_resolution_ws_when_hydration_fails(self):
        event = {
            "id": "event-1",
            "title": "Bitcoin Up or Down",
            "markets": [
                {
                    "id": "market-1",
                    "question": "Bitcoin Up or Down",
                    "clobTokenIds": '["up-token","down-token"]',
                    "outcomes": '["Up","Down"]',
                    "outcomePrices": '["0.495","0.505"]',
                    "startDate": "2026-05-04T00:00:00Z",
                    "endDate": "2026-05-04T00:05:00Z",
                }
            ],
        }

        with patch(
            "custom.btc_agent.market_lookup._MARKET_CACHE",
            {},
        ), patch(
            "custom.btc_agent.market_lookup._SETTLEMENT_THRESHOLD_CACHE",
            {},
        ), patch(
            "custom.btc_agent.market_lookup._fetch_event_by_slug",
            return_value=event,
        ), patch(
            "custom.btc_agent.market_lookup._refresh_market_probabilities",
            side_effect=lambda market: market,
        ), patch(
            "custom.btc_agent.market_lookup._hydrate_missing_threshold_from_page",
            side_effect=lambda market, slug: market,
        ), patch(
            "custom.btc_agent.market_lookup.fetch_btc_resolution_price_for_slug",
            return_value=77763.01,
        ) as mock_fetch_resolution:
            market = get_btc_updown_market_by_slug("btc-updown-5m-1777513800")

        self.assertIsNotNone(market)
        self.assertEqual(market.settlement_threshold, 77763.01)
        mock_fetch_resolution.assert_called_once_with("btc-updown-5m-1777513800")

    def test_fetch_event_by_slug_adds_cache_buster_and_no_cache_headers(self):
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"id": "event-1"}

        with patch(
            "custom.btc_agent.market_lookup.http_get",
            return_value=mock_response,
        ) as mock_http_get, patch(
            "custom.btc_agent.market_lookup.time.time",
            return_value=1777512000.123,
        ), patch(
            "custom.btc_agent.market_lookup.get_polymarket_config",
            return_value=types.SimpleNamespace(gamma_api="https://gamma-api.polymarket.com"),
        ):
            event = _fetch_event_by_slug("btc-updown-5m-1777513800")

        self.assertEqual(event, {"id": "event-1"})
        self.assertEqual(
            mock_http_get.call_args.args[0],
            "https://gamma-api.polymarket.com/events/slug/btc-updown-5m-1777513800",
        )
        self.assertEqual(mock_http_get.call_args.kwargs["params"], {"_ts": 1777512000123})
        self.assertEqual(
            mock_http_get.call_args.kwargs["headers"],
            {
                "cache-control": "no-cache",
                "pragma": "no-cache",
            },
        )

    def test_parse_live_market_probabilities_message_uses_best_ask_from_price_change(self):
        message = json.dumps(
            {
                "market": "0xabc",
                "price_changes": [
                    {
                        "asset_id": "up-token",
                        "price": "0.67",
                        "size": "212.43",
                        "side": "BUY",
                        "best_bid": "0.67",
                        "best_ask": "0.68",
                    },
                    {
                        "asset_id": "down-token",
                        "price": "0.33",
                        "size": "212.43",
                        "side": "SELL",
                        "best_bid": "0.32",
                        "best_ask": "0.33",
                    },
                ],
                "timestamp": "1778122403384",
                "event_type": "price_change",
            }
        )

        up_probability, down_probability = _parse_live_market_probabilities_message(
            message,
            up_token_id="up-token",
            down_token_id="down-token",
        )

        self.assertEqual(up_probability, 0.68)
        self.assertEqual(down_probability, 0.33)

    def test_extract_market_from_event_parses_gamma_outcome_probabilities(self):
        event = {
            "id": "event-1",
            "title": "Bitcoin Up or Down",
            "markets": [
                {
                    "id": "market-1",
                    "question": "Bitcoin Up or Down",
                    "clobTokenIds": '["up-token","down-token"]',
                    "outcomes": '["Up","Down"]',
                    "outcomePrices": '["0.495","0.505"]',
                    "startDate": "2026-05-04T00:00:00Z",
                    "endDate": "2026-05-04T00:05:00Z",
                }
            ],
        }

        market = _extract_market_from_event(event, "btc-updown-5m-1777853100")

        self.assertIsNotNone(market)
        self.assertEqual(market.up_market_probability, 0.495)
        self.assertEqual(market.down_market_probability, 0.505)

    def test_extract_market_from_event_prefers_clob_token_id_order_for_btc_updown(self):
        event = {
            "id": "event-1",
            "title": "Bitcoin Up or Down",
            "markets": [
                {
                    "id": "market-1",
                    "question": "Bitcoin Up or Down",
                    "clobTokenIds": '["up-token-from-order","down-token-from-order"]',
                    "tokens": [
                        {"outcome": "Down", "token_id": "wrong-down-first"},
                        {"outcome": "Up", "token_id": "wrong-up-second"},
                    ],
                    "outcomes": '["Up","Down"]',
                    "outcomePrices": '["0.495","0.505"]',
                    "startDate": "2026-05-04T00:00:00Z",
                    "endDate": "2026-05-04T00:05:00Z",
                }
            ],
        }

        market = _extract_market_from_event(event, "btc-updown-5m-1777853100")

        self.assertIsNotNone(market)
        self.assertEqual(market.up_token_id, "up-token-from-order")
        self.assertEqual(market.down_token_id, "down-token-from-order")

    def test_get_btc_updown_market_by_slug_refreshes_cached_market_probabilities(self):
        cached_market = BtcUpDownMarket(
            event_id="1",
            market_id="2",
            up_token_id="up-token",
            down_token_id="down-token",
            title="Bitcoin Up or Down",
            question="Bitcoin Up or Down",
            slug="manual-slug",
            start_ts=1777056000,
            end_ts=1777056300,
            settlement_threshold=77560.75,
            up_market_probability=0.50,
            down_market_probability=0.50,
        )
        refreshed_event = {
            "id": "1",
            "title": "Bitcoin Up or Down",
            "markets": [
                {
                    "id": "2",
                    "question": "Bitcoin Up or Down",
                    "clobTokenIds": '["up-token","down-token"]',
                    "outcomes": '["Up","Down"]',
                    "outcomePrices": '["0.61","0.39"]',
                    "startDate": "2026-05-04T00:00:00Z",
                    "endDate": "2026-05-04T00:05:00Z",
                }
            ],
        }

        with patch(
            "custom.btc_agent.market_lookup._MARKET_CACHE",
            {"manual-slug": cached_market},
        ), patch(
            "custom.btc_agent.market_lookup._fetch_market_object_by_slug",
            side_effect=Exception("market fetch unavailable"),
        ), patch(
            "custom.btc_agent.market_lookup._fetch_event_by_slug",
            return_value=refreshed_event,
        ) as mock_fetch_event:
            market = get_btc_updown_market_by_slug("manual-slug")

        self.assertEqual(market.settlement_threshold, 77560.75)
        self.assertEqual(market.slug, "manual-slug")
        self.assertEqual(market.up_market_probability, 0.61)
        self.assertEqual(market.down_market_probability, 0.39)
        mock_fetch_event.assert_called_once_with("manual-slug")

    def test_get_btc_updown_market_by_slug_prefers_market_endpoint_for_probability_refresh(self):
        cached_market = BtcUpDownMarket(
            event_id="1",
            market_id="2",
            up_token_id="up-token",
            down_token_id="down-token",
            title="Bitcoin Up or Down",
            question="Bitcoin Up or Down",
            slug="manual-slug",
            start_ts=1777056000,
            end_ts=1777056300,
            settlement_threshold=77560.75,
            up_market_probability=0.50,
            down_market_probability=0.50,
        )
        refreshed_market_object = {
            "id": "2",
            "slug": "btc-updown-5m-1777056000",
            "outcomes": '["Up","Down"]',
            "outcomePrices": '["0.63","0.37"]',
            "startDate": "2026-05-04T00:00:00Z",
            "endDate": "2026-05-04T00:05:00Z",
            "volume": "1234.5",
        }

        with patch(
            "custom.btc_agent.market_lookup._MARKET_CACHE",
            {"manual-slug": cached_market},
        ), patch(
            "custom.btc_agent.market_lookup._fetch_market_object_by_slug",
            return_value=refreshed_market_object,
        ) as mock_fetch_market, patch(
            "custom.btc_agent.market_lookup._fetch_event_by_slug",
        ) as mock_fetch_event:
            market = get_btc_updown_market_by_slug("manual-slug")

        self.assertEqual(market.up_market_probability, 0.63)
        self.assertEqual(market.down_market_probability, 0.37)
        self.assertEqual(market.volume, 1234.5)
        mock_fetch_market.assert_called_once_with("manual-slug")
        mock_fetch_event.assert_not_called()

    def test_get_btc_updown_market_by_slug_prefers_clob_ws_for_btc_probability_refresh(self):
        cached_market = BtcUpDownMarket(
            event_id="1",
            market_id="2",
            up_token_id="up-token",
            down_token_id="down-token",
            title="Bitcoin Up or Down",
            question="Bitcoin Up or Down",
            slug="btc-updown-5m-1777056000",
            start_ts=1777056000,
            end_ts=1777056300,
            settlement_threshold=77560.75,
            up_market_probability=0.50,
            down_market_probability=0.50,
        )

        with patch(
            "custom.btc_agent.market_lookup._MARKET_CACHE",
            {"btc-updown-5m-1777056000": cached_market},
        ), patch(
            "custom.btc_agent.market_lookup.fetch_live_market_probabilities_from_clob_ws",
            return_value=(0.68, 0.33),
        ) as mock_fetch_ws, patch(
            "custom.btc_agent.market_lookup._fetch_market_object_by_slug",
        ) as mock_fetch_market, patch(
            "custom.btc_agent.market_lookup._fetch_event_by_slug",
        ) as mock_fetch_event:
            market = get_btc_updown_market_by_slug("btc-updown-5m-1777056000")

        self.assertEqual(market.up_market_probability, 0.68)
        self.assertEqual(market.down_market_probability, 0.33)

    def test_fetch_live_market_probabilities_from_clob_ws_returns_none_on_handshake_failure(self):
        with patch(
            "custom.btc_agent.market_lookup.websocket.create_connection",
            side_effect=TimeoutError("handshake timed out"),
        ):
            up_probability, down_probability = fetch_live_market_probabilities_from_clob_ws(
                "up-token",
                "down-token",
            )

        self.assertIsNone(up_probability)
        self.assertIsNone(down_probability)

    def test_get_btc_updown_market_by_slug_keeps_cached_btc_probabilities_on_ws_failure(self):
        cached_market = BtcUpDownMarket(
            event_id="1",
            market_id="2",
            up_token_id="up-token",
            down_token_id="down-token",
            title="Bitcoin Up or Down",
            question="Bitcoin Up or Down",
            slug="btc-updown-5m-1777056000",
            start_ts=1777056000,
            end_ts=1777056300,
            settlement_threshold=77560.75,
            up_market_probability=0.50,
            down_market_probability=0.50,
        )

        with patch(
            "custom.btc_agent.market_lookup._MARKET_CACHE",
            {"btc-updown-5m-1777056000": cached_market},
        ), patch(
            "custom.btc_agent.market_lookup.fetch_live_market_probabilities_from_clob_ws",
            return_value=(None, None),
        ) as mock_fetch_ws:
            market = get_btc_updown_market_by_slug("btc-updown-5m-1777056000")

        self.assertEqual(market.up_market_probability, 0.50)
        self.assertEqual(market.down_market_probability, 0.50)
        mock_fetch_ws.assert_called_once_with("up-token", "down-token")

    def test_get_btc_updown_market_by_slug_fetches_live_probabilities_on_initial_btc_lookup(self):
        event = {
            "id": "event-1",
            "title": "Bitcoin Up or Down",
            "markets": [
                {
                    "id": "market-1",
                    "question": "Bitcoin Up or Down",
                    "clobTokenIds": '["up-token","down-token"]',
                    "outcomes": '["Up","Down"]',
                    "outcomePrices": '["0.495","0.505"]',
                    "startDate": "2026-05-04T00:00:00Z",
                    "endDate": "2026-05-04T00:05:00Z",
                    "groupItemThreshold": "77560.75",
                }
            ],
        }

        with patch(
            "custom.btc_agent.market_lookup._MARKET_CACHE",
            {},
        ), patch(
            "custom.btc_agent.market_lookup._fetch_event_by_slug",
            return_value=event,
        ), patch(
            "custom.btc_agent.market_lookup.fetch_live_market_probabilities_from_clob_ws",
            return_value=(0.68, 0.33),
        ) as mock_fetch_live:
            market = get_btc_updown_market_by_slug("btc-updown-5m-1777056000")

        self.assertIsNotNone(market)
        self.assertEqual(market.up_market_probability, 0.68)
        self.assertEqual(market.down_market_probability, 0.33)
        mock_fetch_live.assert_called_once_with("up-token", "down-token")

    def test_extract_market_reads_structured_thresholds_for_btc_updown_markets(self):
        event = {
            "id": "1",
            "title": "Bitcoin Up or Down - April 22, 6:10PM-6:15PM ET",
            "eventMetadata": {
                "priceToBeat": 78842.09031747903,
            },
            "markets": [
                {
                    "id": "2",
                    "question": "Bitcoin Up or Down - April 22, 6:10PM-6:15PM ET",
                    "groupItemThreshold": "0",
                    "tokens": [
                        {"outcome": "Up", "token_id": "up-token"},
                        {"outcome": "Down", "token_id": "down-token"},
                    ],
                    "start_ts": 1776895800,
                    "end_ts": 1776896100,
                }
            ],
        }

        market = _extract_market_from_event(event, "btc-updown-5m-1776895800")

        self.assertIsNotNone(market)
        self.assertEqual(market.settlement_threshold, 78842.09031747903)

    def test_extract_market_prints_gamma_threshold_debug_when_enabled(self):
        event = {
            "id": "1",
            "title": "Bitcoin Up or Down - April 22, 6:10PM-6:15PM ET",
            "eventMetadata": {
                "priceToBeat": 78842.09031747903,
            },
            "markets": [
                {
                    "id": "2",
                    "question": "Bitcoin Up or Down - April 22, 6:10PM-6:15PM ET",
                    "groupItemThreshold": "0",
                    "tokens": [
                        {"outcome": "Up", "token_id": "up-token"},
                        {"outcome": "Down", "token_id": "down-token"},
                    ],
                    "start_ts": 1776895800,
                    "end_ts": 1776896100,
                }
            ],
        }

        with patch(
            "custom.btc_agent.market_lookup.get_trading_config",
            return_value=types.SimpleNamespace(debug=True),
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            market = _extract_market_from_event(event, "btc-updown-5m-1776895800")

        self.assertIsNotNone(market)
        self.assertIn(
            "[DEBUG] Gamma Extract for btc-updown-5m-1776895800: "
            "priceToBeat=78842.09031747903, groupItemThreshold=None -> Result=78842.09031747903",
            stdout.getvalue(),
        )

    def test_extract_market_reads_group_item_threshold_for_btc_updown_markets(self):
        event = {
            "id": "1",
            "title": "Bitcoin Up or Down - April 22, 1:40PM-1:45PM ET",
            "markets": [
                {
                    "id": "2",
                    "question": "Bitcoin Up or Down - April 22, 1:40PM-1:45PM ET",
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
        self.assertEqual(market.question, "Bitcoin Up or Down - April 22, 1:40PM-1:45PM ET")

    def test_extract_market_rejects_unrealistic_small_structured_thresholds(self):
        event = {
            "id": "1",
            "title": "Bitcoin Up or Down - April 23, 3:10PM-3:15PM ET",
            "markets": [
                {
                    "id": "2",
                    "question": "Bitcoin Up or Down - April 23, 3:10PM-3:15PM ET",
                    "groupItemThreshold": 3,
                    "threshold": "3.0",
                    "tokens": [
                        {"outcome": "Up", "token_id": "up-token"},
                        {"outcome": "Down", "token_id": "down-token"},
                    ],
                    "start_ts": 1776971400,
                    "end_ts": 1776971700,
                }
            ],
        }

        market = _extract_market_from_event(event, "btc-updown-5m-1776971400")

        self.assertIsNotNone(market)
        self.assertIsNone(market.settlement_threshold)

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

    def test_parse_threshold_from_text_ignores_time_of_day(self):
        threshold = _parse_threshold_from_text(
            "Bitcoin Up or Down - April 23, 3:00PM-3:05PM ET"
        )

        self.assertIsNone(threshold)

    def test_extract_market_parses_iso_market_times(self):
        event = {
            "id": "1",
            "title": "Bitcoin Up or Down - April 23, 4:00PM-4:05PM ET",
            "markets": [
                {
                    "id": "2",
                    "question": "Bitcoin Up or Down - April 23, 4:00PM-4:05PM ET",
                    "eventStartTime": "2026-04-23T23:00:00Z",
                    "endDate": "2026-04-23T23:05:00Z",
                    "tokens": [
                        {"outcome": "Up", "token_id": "up-token"},
                        {"outcome": "Down", "token_id": "down-token"},
                    ],
                }
            ],
        }

        market = _extract_market_from_event(event, "btc-updown-5m-1776985200")

        self.assertIsNotNone(market)
        self.assertEqual(market.start_ts, 1776985200)
        self.assertEqual(market.end_ts, 1776985500)

    def test_extract_event_from_next_data_returns_slug_query_event(self):
        payload = {
            "props": {
                "pageProps": {
                    "dehydratedState": {
                        "queries": [
                            {
                                "queryKey": ["/api/event/slug", "btc-updown-5m-1776897000"],
                                "state": {
                                    "data": {
                                        "id": "event-1",
                                        "slug": "btc-updown-5m-1776897000",
                                        "eventMetadata": {"priceToBeat": 78564.68601198489},
                                    }
                                },
                            }
                        ]
                    }
                }
            }
        }

        event = _extract_event_from_next_data(payload, "btc-updown-5m-1776897000")

        self.assertIsNotNone(event)
        self.assertEqual(event["eventMetadata"]["priceToBeat"], 78564.68601198489)

    def test_extract_event_from_next_data_returns_page_props_event(self):
        payload = {
            "props": {
                "pageProps": {
                    "event": {
                        "id": "event-2",
                        "slug": "btc-updown-5m-1776972900",
                        "title": "Bitcoin Up or Down",
                        "eventMetadata": {"priceToBeat": 77731.41317476261},
                        "markets": [
                            {
                                "id": "market-2",
                                "question": "Bitcoin Up or Down",
                                "tokens": [
                                    {"outcome": "Up", "token_id": "up-token"},
                                    {"outcome": "Down", "token_id": "down-token"},
                                ],
                            }
                        ],
                    }
                }
            }
        }

        event = _extract_event_from_next_data(payload, "btc-updown-5m-1776972900")

        self.assertIsNotNone(event)
        self.assertEqual(event["eventMetadata"]["priceToBeat"], 77731.41317476261)

    def test_extract_event_from_next_data_returns_top_level_page_props_event(self):
        payload = {
            "pageProps": {
                "event": {
                    "id": "event-3",
                    "slug": "btc-updown-5m-1776972900",
                    "title": "Bitcoin Up or Down",
                    "eventMetadata": {"priceToBeat": 77731.41317476261},
                    "markets": [
                        {
                            "id": "market-3",
                            "question": "Bitcoin Up or Down",
                            "tokens": [
                                {"outcome": "Up", "token_id": "up-token"},
                                {"outcome": "Down", "token_id": "down-token"},
                            ],
                        }
                    ],
                }
            }
        }

        event = _extract_event_from_next_data(payload, "btc-updown-5m-1776972900")

        self.assertIsNotNone(event)
        self.assertEqual(event["eventMetadata"]["priceToBeat"], 77731.41317476261)

    def test_extract_next_build_id_parses_next_data_html(self):
        html = """
        <script id="__NEXT_DATA__" type="application/json">
        {"buildId":"build-TfctsWXpff2fKS","props":{"pageProps":{}}}
        </script>
        """

        build_id = _extract_next_build_id(html)

        self.assertEqual(build_id, "build-TfctsWXpff2fKS")

    def test_extract_next_build_id_parses_crossorigin_next_data_html(self):
        html = """
        <script id="__NEXT_DATA__" type="application/json" crossorigin="anonymous">
        {"buildId":"build-TfctsWXpff2fKS","pageProps":{}}
        </script>
        """

        build_id = _extract_next_build_id(html)

        self.assertEqual(build_id, "build-TfctsWXpff2fKS")

    def test_extract_embedded_next_data_payload_parses_html_script_payload(self):
        html = """
        <script id="__NEXT_DATA__" type="application/json">
        {"buildId":"build-TfctsWXpff2fKS","pageProps":{"event":{"id":"1"}}}
        </script>
        """

        payload = _extract_embedded_next_data_payload(html)

        self.assertEqual(payload["buildId"], "build-TfctsWXpff2fKS")
        self.assertEqual(payload["pageProps"]["event"]["id"], "1")

    def test_fetch_event_from_next_data_route_parses_next_json_payload(self):
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {
            "props": {
                "pageProps": {
                    "event": {
                        "id": "event-2",
                        "slug": "btc-updown-5m-1776972900",
                        "eventMetadata": {"priceToBeat": 77731.41317476261},
                    }
                }
            }
        }

        with patch(
            "custom.btc_agent.market_lookup.http_get",
            return_value=mock_response,
        ):
            event = _fetch_event_from_next_data_route(
                "btc-updown-5m-1776972900",
                "build-TfctsWXpff2fKS",
            )

        self.assertIsNotNone(event)
        self.assertEqual(event["eventMetadata"]["priceToBeat"], 77731.41317476261)

    def test_fetch_next_data_payload_returns_full_json(self):
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"pageProps": {"key": '["btc-updown-5m-1776979200"]'}}

        with patch(
            "custom.btc_agent.market_lookup.http_get",
            return_value=mock_response,
        ):
            payload = _fetch_next_data_payload(
                "btc-updown-5m-1776979200",
                "build-TfctsWXpff2fKS",
            )

        self.assertEqual(payload["pageProps"]["key"], '["btc-updown-5m-1776979200"]')

    def test_extract_previous_period_close_from_next_data_uses_matching_end_time(self):
        payload = {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "state": {
                                "data": {
                                    "data": {
                                        "results": [
                                            {
                                                "startTime": "2026-04-23T21:10:00.000Z",
                                                "endTime": "2026-04-23T21:15:00.000Z",
                                                "openPrice": 77858.8682480781,
                                                "closePrice": 77903.23396,
                                                "outcome": "up",
                                                "percentChange": 0.05698222042033588,
                                            },
                                            {
                                                "startTime": "2026-04-23T21:15:00.000Z",
                                                "endTime": "2026-04-23T21:20:00.000Z",
                                                "openPrice": 77903.23396,
                                                "closePrice": 77885.34596,
                                                "outcome": "down",
                                                "percentChange": -0.02296181954291713,
                                            },
                                            {
                                                "startTime": "2026-04-23T21:20:00.000Z",
                                                "endTime": "2026-04-23T21:25:00.000Z",
                                                "openPrice": 77885.34596,
                                                "closePrice": 77867.371,
                                                "outcome": "down",
                                                "percentChange": -0.023078744503797065,
                                            },
                                        ]
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }

        threshold = _extract_previous_period_close_from_next_data(
            payload,
            "btc-updown-5m-1776979200",
        )

        self.assertEqual(threshold, 77885.34596)

    def test_extract_previous_period_close_from_next_data_falls_back_to_latest_prior_close(self):
        payload = {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "state": {
                                "data": {
                                    "data": {
                                        "results": [
                                            {
                                                "startTime": "2026-04-29T19:00:00.000Z",
                                                "endTime": "2026-04-29T19:05:00.000Z",
                                                "openPrice": 75520.1000000000,
                                                "closePrice": 75491.41106368953,
                                                "outcome": "down",
                                                "percentChange": -0.0379,
                                            },
                                            {
                                                "startTime": "2026-04-29T19:05:00.000Z",
                                                "endTime": "2026-04-29T19:10:00.000Z",
                                                "openPrice": 75491.41106368953,
                                                "closePrice": 75374.81761843128,
                                                "outcome": "down",
                                                "percentChange": -0.15444597420479106,
                                            },
                                        ]
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }

        threshold = _extract_previous_period_close_from_next_data(
            payload,
            "btc-updown-5m-1777490100",
        )

        self.assertEqual(threshold, 75374.81761843128)

    def test_extract_previous_period_close_from_next_data_exact_only_does_not_use_older_fallback(self):
        payload = {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "state": {
                                "data": {
                                    "data": {
                                        "results": [
                                            {
                                                "startTime": "2026-04-29T19:05:00.000Z",
                                                "endTime": "2026-04-29T19:10:00.000Z",
                                                "openPrice": 75491.41106368953,
                                                "closePrice": 75374.81761843128,
                                                "outcome": "down",
                                                "percentChange": -0.15444597420479106,
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }

        threshold = _extract_previous_period_close_from_next_data(
            payload,
            "btc-updown-5m-1777490100",
            allow_latest_prior_fallback=False,
        )

        self.assertIsNone(threshold)

    def test_extract_previous_period_final_price_from_next_data_uses_matching_end_date(self):
        payload = {
            "pageProps": {
                "event": {
                    "markets": [
                        {
                            "endDate": "2026-04-24T18:45:00Z",
                            "eventMetadata": {
                                "finalPrice": 77598.79949998436,
                                "priceToBeat": 77560.75,
                            },
                        }
                    ]
                }
            }
        }

        threshold = _extract_previous_period_final_price_from_next_data(
            payload,
            "btc-updown-5m-1777056300",
            allow_latest_prior_fallback=False,
        )

        self.assertEqual(threshold, 77598.79949998436)

    def test_extract_live_period_open_from_next_data_uses_crypto_prices_query(self):
        payload = {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "queryKey": [
                                "crypto-prices",
                                "price",
                                "BTC",
                                "2026-04-23T21:40:00Z",
                                "fiveminute",
                                "2026-04-23T21:45:00Z",
                            ],
                            "state": {
                                "data": {
                                    "openPrice": 78019.41,
                                    "closePrice": None,
                                }
                            },
                        }
                    ]
                }
            }
        }

        threshold = _extract_live_period_open_from_next_data(
            payload,
            "btc-updown-5m-1776980400",
        )

        self.assertEqual(threshold, 78019.41)

    def test_extract_live_period_open_from_next_data_uses_null_close_state_fallback(self):
        payload = {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "state": {
                                "data": {
                                    "openPrice": 75763.00543733485,
                                    "closePrice": None,
                                }
                            },
                        },
                        {
                            "state": {
                                "data": {
                                    "data": {
                                        "results": [
                                            {
                                                "startTime": "2026-04-29T21:10:00.000Z",
                                                "endTime": "2026-04-29T21:15:00.000Z",
                                                "openPrice": 75655.21496328014,
                                                "closePrice": 75763.00543733485,
                                                "outcome": "up",
                                                "percentChange": 0.14247593388905994,
                                            }
                                        ]
                                    }
                                }
                            },
                        },
                    ]
                }
            }
        }

        threshold = _extract_live_period_open_from_next_data(
            payload,
            "btc-updown-5m-1777497300",
        )

        self.assertIsNone(threshold)

    def test_extract_current_period_open_from_next_data_uses_null_close_state_fallback(self):
        payload = {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "state": {
                                "data": {
                                    "openPrice": 75763.00543733485,
                                    "closePrice": None,
                                }
                            },
                        },
                        {
                            "state": {
                                "data": {
                                    "data": {
                                        "results": [
                                            {
                                                "startTime": "2026-04-29T21:10:00.000Z",
                                                "endTime": "2026-04-29T21:15:00.000Z",
                                                "openPrice": 75655.21496328014,
                                                "closePrice": 75763.00543733485,
                                                "outcome": "up",
                                                "percentChange": 0.14247593388905994,
                                            }
                                        ]
                                    }
                                }
                            },
                        },
                    ]
                }
            }
        }

        threshold = _extract_current_period_open_from_next_data(
            payload,
            "btc-updown-5m-1777497300",
        )

        self.assertEqual(threshold, 75763.00543733485)

    def test_build_price_to_beat_debug_report_includes_curl_and_live_open(self):
        payload = {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "queryKey": [
                                "crypto-prices",
                                "price",
                                "BTC",
                                "2026-04-23T22:25:00Z",
                                "fiveminute",
                                "2026-04-23T22:30:00Z",
                            ],
                            "state": {
                                "data": {
                                    "openPrice": 78218.01972274295,
                                    "closePrice": None,
                                }
                            },
                        }
                    ]
                }
            }
        }

        with patch(
            "custom.btc_agent.market_lookup._fetch_polymarket_page",
            return_value='<script id="__NEXT_DATA__" type="application/json">{"buildId":"build-TfctsWXpff2fKS"}</script>',
        ), patch(
            "custom.btc_agent.market_lookup._fetch_next_data_payload_chain",
            return_value=[("build-TfctsWXpff2fKS", payload)],
        ):
            report = build_price_to_beat_debug_report("btc-updown-5m-1776983100")
            reports = build_price_to_beat_debug_reports("btc-updown-5m-1776983100")
            next_data_report = reports[1]

        self.assertIn("next_data_curl=curl 'https://polymarket.com/_next/data/build-TfctsWXpff2fKS/en/event/btc-updown-5m-1776983100.json?slug=btc-updown-5m-1776983100'", report)
        self.assertIn("live_period_open=78218.01972274295", next_data_report)
        self.assertIn("current_period_open=78218.01972274295", next_data_report)

    def test_build_price_to_beat_debug_reports_splits_embedded_and_next_data_payloads(self):
        embedded_html = (
            '<script id="__NEXT_DATA__" type="application/json">'
            '{"buildId":"build-TfctsWXpff2fKS","pageProps":{"dehydratedState":{"queries":[{"state":{"data":{"openPrice":75763.00543733485,"closePrice":null}}}]}}}'
            "</script>"
        )
        next_data_payload = {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "queryKey": [
                                "crypto-prices",
                                "price",
                                "BTC",
                                "2026-04-29T21:15:00Z",
                                "fiveminute",
                                "2026-04-29T21:20:00Z",
                            ],
                            "state": {
                                "data": {
                                    "openPrice": 75763.00543733485,
                                    "closePrice": None,
                                }
                            },
                        }
                    ]
                }
            }
        }

        with patch(
            "custom.btc_agent.market_lookup._fetch_polymarket_page",
            return_value=embedded_html,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_next_data_payload_chain",
            return_value=[("build-TfctsWXpff2fKS", next_data_payload)],
        ):
            reports = build_price_to_beat_debug_reports(
                "btc-updown-5m-1777497300"
            )
            page_report = reports[0]
            next_data_report = reports[1]

        self.assertIn("embedded_page_payload=", page_report)
        self.assertIn('"openPrice": 75763.00543733485', page_report)
        self.assertIn("next_data_payload=", next_data_report)
        self.assertIn("next_data_fetch=success", next_data_report)

    def test_build_price_to_beat_debug_reports_emits_multiple_next_data_pages(self):
        embedded_html = """
        <script id="__NEXT_DATA__" type="application/json">
        {"buildId":"build-one","pageProps":{"event":{"id":"1"}}}
        </script>
        """
        payload_one = {"buildId": "build-two", "pageProps": {"dehydratedState": {"queries": []}}}
        payload_two = {"pageProps": {"dehydratedState": {"queries": []}}}

        with patch(
            "custom.btc_agent.market_lookup._fetch_polymarket_page",
            return_value=embedded_html,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_next_data_payload_chain",
            return_value=[("build-one", payload_one), ("build-two", payload_two)],
        ):
            reports = build_price_to_beat_debug_reports("btc-updown-5m-1777497300")

        self.assertEqual(len(reports), 3)
        self.assertIn("build_id=build-one", reports[1])
        self.assertIn("build_id=build-two", reports[2])

    def test_fetch_next_data_payload_chain_repeats_same_build_id_across_pages(self):
        payload = {"pageProps": {"dehydratedState": {"queries": []}}}

        with patch(
            "custom.btc_agent.market_lookup._fetch_next_data_payload",
            side_effect=[payload, payload, payload],
        ) as mock_fetch, patch(
            "custom.btc_agent.market_lookup.time.sleep",
        ):
            pages = _fetch_next_data_payload_chain(
                "btc-updown-5m-1777503900",
                "build-TfctsWXpff2fKS",
                max_pages=3,
            )

        self.assertEqual(len(pages), 3)
        self.assertEqual(
            pages,
            [
                ("build-TfctsWXpff2fKS", payload),
                ("build-TfctsWXpff2fKS", payload),
                ("build-TfctsWXpff2fKS", payload),
            ],
        )
        self.assertEqual(mock_fetch.call_count, 3)
        self.assertEqual(
            mock_fetch.call_args_list[0].kwargs["request_number"],
            1,
        )
        self.assertEqual(
            mock_fetch.call_args_list[1].kwargs["request_number"],
            2,
        )
        self.assertEqual(
            mock_fetch.call_args_list[2].kwargs["request_number"],
            3,
        )

    def test_fetch_next_data_payload_uses_cache_busting_query_params_and_headers(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"pageProps": {}}
        response.raise_for_status = Mock()

        with patch(
            "custom.btc_agent.market_lookup.http_get",
            return_value=response,
        ) as mock_http_get, patch(
            "custom.btc_agent.market_lookup.time.time",
            return_value=1777512000.123,
        ):
            payload = _fetch_next_data_payload(
                "btc-updown-5m-1777511400",
                "build-TfctsWXpff2fKS",
                request_number=2,
            )

        self.assertEqual(payload, {"pageProps": {}})
        mock_http_get.assert_called_once()
        self.assertEqual(
            mock_http_get.call_args.args[0],
            "https://polymarket.com/_next/data/build-TfctsWXpff2fKS/en/event/btc-updown-5m-1777511400.json",
        )
        self.assertEqual(
            mock_http_get.call_args.kwargs["params"],
            {
                "slug": "btc-updown-5m-1777511400",
                "_req": 2,
                "_ts": 1777512000123,
            },
        )
        self.assertEqual(
            mock_http_get.call_args.kwargs["headers"],
            {
                "accept": "*/*",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "x-nextjs-data": "1",
            },
        )

    def test_write_current_period_dataset_file_writes_current_period_json(self):
        dataset = {
            "slug": "btc-updown-5m-1777503900",
            "selected_next_data_pages": [],
        }

        with patch(
            "custom.btc_agent.market_lookup.os.getcwd",
            return_value="/appl/agents",
        ):
            _write_current_period_dataset_file(dataset)

        with open("/appl/agents/data_files/current_period.json", encoding="utf-8") as data_file:
            self.assertIn('"slug": "btc-updown-5m-1777503900"', data_file.read())

    def test_build_current_period_dataset_keeps_first_three_requests(self):
        dataset = _build_current_period_dataset(
            slug="btc-updown-5m-1777503900",
            html="",
            embedded_payload=None,
            build_id="build-TfctsWXpff2fKS",
            payload_chain=[
                ("build-TfctsWXpff2fKS", {"page": 1}),
                ("build-TfctsWXpff2fKS", {"page": 2}),
                ("build-TfctsWXpff2fKS", {"page": 3}),
            ],
        )

        self.assertEqual(
            dataset["selected_next_data_pages"],
            [
                {
                    "request_number": 1,
                    "build_id": "build-TfctsWXpff2fKS",
                    "next_data_url": "https://polymarket.com/_next/data/build-TfctsWXpff2fKS/en/event/btc-updown-5m-1777503900.json?slug=btc-updown-5m-1777503900",
                    "payload": {"page": 1},
                },
                {
                    "request_number": 2,
                    "build_id": "build-TfctsWXpff2fKS",
                    "next_data_url": "https://polymarket.com/_next/data/build-TfctsWXpff2fKS/en/event/btc-updown-5m-1777503900.json?slug=btc-updown-5m-1777503900",
                    "payload": {"page": 2},
                },
                {
                    "request_number": 3,
                    "build_id": "build-TfctsWXpff2fKS",
                    "next_data_url": "https://polymarket.com/_next/data/build-TfctsWXpff2fKS/en/event/btc-updown-5m-1777503900.json?slug=btc-updown-5m-1777503900",
                    "payload": {"page": 3},
                },
            ],
        )

    def test_hydrate_missing_threshold_prints_debug_api_exceptions(self):
        market = BtcUpDownMarket(
            event_id="1",
            market_id="2",
            up_token_id="up-token",
            down_token_id="down-token",
            title="Bitcoin Up or Down",
            question="Bitcoin Up or Down",
            slug="btc-updown-5m-1777490100",
            start_ts=1777490100,
            end_ts=1777490400,
            settlement_threshold=None,
        )

        with patch(
            "custom.btc_agent.market_lookup._fetch_vatic_price_to_beat_by_slug",
            side_effect=RuntimeError("vatic down"),
        ), patch(
            "custom.btc_agent.market_lookup._fetch_price_to_beat_by_slug",
            side_effect=RuntimeError("p2b down"),
        ), patch(
            "custom.btc_agent.market_lookup._fetch_price_via_selenium",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup.get_trading_config",
            return_value=types.SimpleNamespace(debug=True),
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            hydrated = _hydrate_missing_threshold_from_page(market, market.slug)

        self.assertIs(hydrated, market)
        self.assertIn("[DEBUG] Vatic API Exception: vatic down", stdout.getvalue())
        self.assertIn("[DEBUG] Polymarket P2B API Exception: p2b down", stdout.getvalue())
        self.assertIn(
            "[DEBUG] Vatic and Polymarket API failed, attempting Selenium scrape for btc-updown-5m-1777490100...",
            stdout.getvalue(),
        )

    def test_hydrate_missing_threshold_prefers_polymarket_api_after_vatic_miss(self):
        market = BtcUpDownMarket(
            event_id="1",
            market_id="2",
            up_token_id="up-token",
            down_token_id="down-token",
            title="Bitcoin Up or Down",
            question="Bitcoin Up or Down",
            slug="btc-updown-5m-1777493100",
            start_ts=1777493100,
            end_ts=1777493400,
            settlement_threshold=None,
        )

        with patch(
            "custom.btc_agent.market_lookup._fetch_vatic_price_to_beat_by_slug",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_price_to_beat_by_slug",
            return_value=75600.465495,
        ), patch("custom.btc_agent.market_lookup._fetch_price_via_selenium") as mock_selenium:
            hydrated_market = _hydrate_missing_threshold_from_page(
                market,
                "btc-updown-5m-1777493100",
            )

        self.assertEqual(hydrated_market.settlement_threshold, 75600.465495)
        mock_selenium.assert_not_called()

    def test_hydrate_missing_threshold_uses_selenium_after_api_misses(self):
        market = BtcUpDownMarket(
            event_id="1",
            market_id="2",
            up_token_id="up-token",
            down_token_id="down-token",
            title="Bitcoin Up or Down",
            question="Bitcoin Up or Down",
            slug="btc-updown-5m-1777496400",
            start_ts=1777496400,
            end_ts=1777496700,
            settlement_threshold=None,
        )

        with patch(
            "custom.btc_agent.market_lookup._fetch_vatic_price_to_beat_by_slug",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_price_to_beat_by_slug",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_price_via_selenium",
            return_value=75855.07855,
        ):
            hydrated_market = _hydrate_missing_threshold_from_page(
                market,
                "btc-updown-5m-1777496400",
            )

        self.assertEqual(hydrated_market.settlement_threshold, 75855.07855)

    def test_hydrate_missing_threshold_prefers_vatic_price_for_btc_updown_markets(self):
        market = BtcUpDownMarket(
            event_id="1",
            market_id="2",
            up_token_id="up-token",
            down_token_id="down-token",
            title="Bitcoin Up or Down - April 29, 9:50PM-9:55PM ET",
            question="Bitcoin Up or Down - April 29, 9:50PM-9:55PM ET",
            slug="btc-updown-5m-1777513800",
            start_ts=1777513800,
            end_ts=1777514100,
            settlement_threshold=None,
        )

        with patch(
            "custom.btc_agent.market_lookup._fetch_vatic_price_to_beat_by_slug",
            return_value=77761.01,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_price_to_beat_by_slug",
        ) as mock_fetch_api, patch(
            "custom.btc_agent.market_lookup._fetch_price_via_selenium",
        ) as mock_selenium:
            hydrated_market = _hydrate_missing_threshold_from_page(
                market,
                "btc-updown-5m-1777513800",
            )

        self.assertEqual(hydrated_market.settlement_threshold, 77761.01)
        mock_fetch_api.assert_not_called()
        mock_selenium.assert_not_called()

    def test_fetch_price_via_selenium_returns_none_when_dependencies_missing(self):
        with patch("custom.btc_agent.market_lookup.webdriver", None), patch(
            "custom.btc_agent.market_lookup.get_trading_config",
            return_value=types.SimpleNamespace(debug=True),
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            threshold = _fetch_price_via_selenium("btc-updown-5m-1777513800")

        self.assertIsNone(threshold)
        self.assertIn(
            "[DEBUG] Selenium unavailable for btc-updown-5m-1777513800",
            stdout.getvalue(),
        )

    def test_extract_threshold_from_price_to_beat_response_handles_nested_payload(self):
        threshold = _extract_threshold_from_price_to_beat_response(
            {
                "data": {
                    "priceToBeat": "77722.39",
                }
            }
        )

        self.assertEqual(threshold, 77722.39)

    def test_extract_vatic_price_from_response_handles_nested_payload(self):
        threshold = _extract_vatic_price_from_response(
            {
                "data": {
                    "target": {
                        "price": "77722.39",
                    }
                }
            }
        )

        self.assertEqual(threshold, 77722.39)

    def test_fetch_price_to_beat_by_slug_parses_direct_api_response(self):
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"priceToBeat": "77722.39"}

        with patch(
            "custom.btc_agent.market_lookup.http_get",
            return_value=mock_response,
        ) as mock_http_get, patch(
            "custom.btc_agent.market_lookup.time.time",
            return_value=1777512000.123,
        ):
            threshold = _fetch_price_to_beat_by_slug("tesla-up-or-down")

        self.assertEqual(threshold, 77722.39)
        mock_http_get.assert_called_once_with(
            "https://polymarket.com/api/equity/price-to-beat/tesla-up-or-down",
            params={"_ts": 1777512000123},
            headers={
                "cache-control": "no-cache",
                "pragma": "no-cache",
            },
            timeout=10,
        )

    def test_fetch_vatic_price_to_beat_by_slug_parses_price_response(self):
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"price": 77763.01}

        with patch(
            "custom.btc_agent.market_lookup.http_get",
            return_value=mock_response,
        ) as mock_http_get, patch(
            "custom.btc_agent.market_lookup.time.time",
            return_value=1777512000.123,
        ):
            threshold = _fetch_vatic_price_to_beat_by_slug("btc-updown-5m-1777513800")

        self.assertEqual(threshold, 77763.01)
        self.assertEqual(
            mock_http_get.call_args.kwargs["params"],
            {
                "asset": "btc",
                "type": "5min",
                "timestamp": "1777513800",
                "_ts": 1777512000123,
            },
        )
        self.assertEqual(
            mock_http_get.call_args.kwargs["headers"],
            {
                "cache-control": "no-cache",
                "pragma": "no-cache",
            },
        )

    def test_fetch_vatic_price_to_beat_by_slug_prints_debug_response(self):
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"price": 77763.01}

        with patch(
            "custom.btc_agent.market_lookup.http_get",
            return_value=mock_response,
        ), patch(
            "custom.btc_agent.market_lookup.get_trading_config",
            return_value=types.SimpleNamespace(debug=True),
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            threshold = _fetch_vatic_price_to_beat_by_slug("btc-updown-5m-1777513800")

        self.assertEqual(threshold, 77763.01)
        self.assertIn('[DEBUG] Vatic API Response: {"price": 77763.01}', stdout.getvalue())

    def test_fetch_price_to_beat_by_slug_tries_standard_endpoint_for_btc_updown_market_slugs(self):
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"priceToBeat": "77722.39"}

        with patch("custom.btc_agent.market_lookup.http_get", return_value=mock_response) as mock_http_get, patch(
            "custom.btc_agent.market_lookup.time.time",
            return_value=1777512000.123,
        ):
            threshold = _fetch_price_to_beat_by_slug("btc-updown-5m-1776971400")

        self.assertEqual(threshold, 77722.39)
        mock_http_get.assert_called_once_with(
            "https://polymarket.com/api/equity/price-to-beat/btc-updown-5m-1776971400",
            params={"_ts": 1777512000123},
            headers={
                "cache-control": "no-cache",
                "pragma": "no-cache",
            },
            timeout=10,
        )

    def test_fetch_price_to_beat_by_slug_prints_debug_response(self):
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"priceToBeat": "77722.39"}

        with patch(
            "custom.btc_agent.market_lookup.http_get",
            return_value=mock_response,
        ), patch(
            "custom.btc_agent.market_lookup.get_trading_config",
            return_value=types.SimpleNamespace(debug=True),
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            threshold = _fetch_price_to_beat_by_slug("btc-updown-5m-1776971400")

        self.assertEqual(threshold, 77722.39)
        self.assertIn(
            '[DEBUG] Polymarket P2B API Response: {"priceToBeat": "77722.39"}',
            stdout.getvalue(),
        )

    def test_extract_threshold_from_page_html_parses_price_to_beat_label(self):
        html = """
        <div>
            Each market page shows the live Price to Beat ($69,498.91)
            and the current live Bitcoin price.
        </div>
        """

        threshold = _extract_threshold_from_page_html(html)

        self.assertEqual(threshold, 69498.91)

    def test_extract_threshold_from_page_html_parses_faq_price_to_beat_text(self):
        html = """
        <div>
            To trade on this market, decide whether you believe Bitcoin's price
            will finish above or below the opening "Price to Beat" of $77,722.39 by 3:15PM ET.
        </div>
        """

        threshold = _extract_threshold_from_page_html(html)

        self.assertEqual(threshold, 77722.39)

    def test_extract_threshold_from_page_html_parses_inspector_dom_snippet(self):
        html = """
        <div class="flex items-center gap-1 justify-between">
            <span class="text-body-xs font-semibold" style="color: var(--color-text-secondary); opacity: 0.8;">
                Price To Beat
            </span>
        </div>
        <span class="mt-1 tracking-wide font-[620] text-text-secondary text-heading-2xl">
            $77,722.39
        </span>
        """

        threshold = _extract_threshold_from_page_html(html)

        self.assertEqual(threshold, 77722.39)

    def test_extract_threshold_from_page_html_prefers_labeled_text_heading_span(self):
        html = """
        <div class="flex items-center gap-1 justify-between">
            <span class="text-body-xs font-semibold">Price To Beat</span>
        </div>
        <div>ignore 1 and 5 here</div>
        <span class="mt-1 tracking-wide font-[620] text-text-secondary text-heading-2xl">$77,722.39</span>
        """

        threshold = _extract_threshold_from_page_html(html)

        self.assertEqual(threshold, 77722.39)

    def test_extract_threshold_from_page_html_parses_direct_text_heading_span(self):
        html = """
        <div>other values like 1 and 5 should be ignored</div>
        <span class="mt-1 tracking-wide font-[620] text-text-secondary text-heading-2xl">$77,722.39</span>
        """

        threshold = _extract_threshold_from_page_html(html)

        self.assertEqual(threshold, 77722.39)

    def test_extract_threshold_from_page_html_rejects_small_direct_text_heading_span(self):
        html = """
        <span class="mt-1 tracking-wide font-[620] text-text-secondary text-heading-2xl">$1.00</span>
        """

        threshold = _extract_threshold_from_page_html(html)

        self.assertIsNone(threshold)


if __name__ == "__main__":
    unittest.main()
