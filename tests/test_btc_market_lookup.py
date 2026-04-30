import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))

from custom.btc_agent.market_lookup import (
    BtcUpDownMarket,
    _build_current_period_dataset,
    build_price_to_beat_debug_report,
    build_price_to_beat_debug_reports,
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
    _write_current_period_dataset_file,
    _extract_threshold_from_page_html,
    _parse_threshold_from_text,
)


class TestBtcMarketLookup(unittest.TestCase):
    def test_get_btc_updown_market_by_slug_returns_cached_market_without_refetch(self):
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
        )

        with patch(
            "custom.btc_agent.market_lookup._MARKET_CACHE",
            {"btc-updown-5m-1777056000": cached_market},
        ), patch(
            "custom.btc_agent.market_lookup._fetch_event_by_slug",
        ) as mock_fetch_event:
            market = get_btc_updown_market_by_slug("btc-updown-5m-1777056000")

        self.assertEqual(market.settlement_threshold, 77560.75)
        self.assertEqual(market.slug, "btc-updown-5m-1777056000")
        mock_fetch_event.assert_not_called()

    def test_extract_market_ignores_structured_thresholds_for_btc_updown_markets(self):
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
        self.assertIsNone(market.settlement_threshold)

    def test_extract_market_ignores_group_item_threshold_for_btc_updown_markets(self):
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
        self.assertIsNone(market.settlement_threshold)
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

    def test_hydrate_missing_threshold_prefers_live_period_open_after_retry(self):
        market = BtcUpDownMarket(
            event_id="1",
            market_id="2",
            up_token_id="up-token",
            down_token_id="down-token",
            title="Bitcoin Up or Down - April 29, 3:15PM-3:20PM ET",
            question="Bitcoin Up or Down - April 29, 3:15PM-3:20PM ET",
            slug="btc-updown-5m-1777490100",
            start_ts=1777490100,
            end_ts=1777490400,
            settlement_threshold=None,
        )
        first_payload = {
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
        second_payload = {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "queryKey": [
                                "crypto-prices",
                                "price",
                                "BTC",
                                "2026-04-29T19:15:00Z",
                                "fiveminute",
                                "2026-04-29T19:20:00Z",
                            ],
                            "state": {
                                "data": {
                                    "openPrice": 75403.56802142781,
                                    "closePrice": None,
                                }
                            },
                        }
                    ]
                }
            }
        }

        with patch(
            "custom.btc_agent.market_lookup._fetch_vatic_price_to_beat_by_slug",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_price_to_beat_by_slug",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_polymarket_page",
            return_value='<script id="__NEXT_DATA__" type="application/json">{"buildId":"build-TfctsWXpff2fKS"}</script>',
        ), patch(
            "custom.btc_agent.market_lookup._fetch_next_data_payload_chain",
            side_effect=[
                [("build-TfctsWXpff2fKS", first_payload)],
                [("build-TfctsWXpff2fKS", second_payload)],
            ],
        ), patch(
            "custom.btc_agent.market_lookup.time.sleep",
        ):
            hydrated_market = _hydrate_missing_threshold_from_page(
                market,
                "btc-updown-5m-1777490100",
            )

        self.assertEqual(hydrated_market.settlement_threshold, 75403.56802142781)

    def test_hydrate_missing_threshold_fetches_next_data_when_embedded_payload_lacks_value(self):
        market = BtcUpDownMarket(
            event_id="1",
            market_id="2",
            up_token_id="up-token",
            down_token_id="down-token",
            title="Bitcoin Up or Down - April 29, 4:05PM-4:10PM ET",
            question="Bitcoin Up or Down - April 29, 4:05PM-4:10PM ET",
            slug="btc-updown-5m-1777493100",
            start_ts=1777493100,
            end_ts=1777493400,
            settlement_threshold=None,
        )
        embedded_html = """
        <script id="__NEXT_DATA__" type="application/json">
        {"buildId":"build-TfctsWXpff2fKS","pageProps":{"event":{"id":"1"}}}
        </script>
        """
        next_data_payload = {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "queryKey": [
                                "crypto-prices",
                                "price",
                                "BTC",
                                "2026-04-29T20:05:00Z",
                                "fiveminute",
                                "2026-04-29T20:10:00Z",
                            ],
                            "state": {
                                "data": {
                                    "openPrice": 75600.465495,
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
                                                "startTime": "2026-04-29T20:00:00.000Z",
                                                "endTime": "2026-04-29T20:05:00.000Z",
                                                "openPrice": 75580.125,
                                                "closePrice": 75529.57330485078,
                                                "outcome": "down",
                                                "percentChange": -0.0669,
                                            }
                                        ]
                                    }
                                }
                            }
                        },
                    ]
                }
            }
        }

        with patch(
            "custom.btc_agent.market_lookup._fetch_vatic_price_to_beat_by_slug",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_price_to_beat_by_slug",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_polymarket_page",
            return_value=embedded_html,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_next_data_payload_chain",
            return_value=[("build-TfctsWXpff2fKS", next_data_payload)],
        ):
            hydrated_market = _hydrate_missing_threshold_from_page(
                market,
                "btc-updown-5m-1777493100",
            )

        self.assertEqual(hydrated_market.settlement_threshold, 75600.465495)

    def test_hydrate_missing_threshold_uses_prior_close_when_open_sources_missing(self):
        market = BtcUpDownMarket(
            event_id="1",
            market_id="2",
            up_token_id="up-token",
            down_token_id="down-token",
            title="Bitcoin Up or Down - April 29, 5:00PM-5:05PM ET",
            question="Bitcoin Up or Down - April 29, 5:00PM-5:05PM ET",
            slug="btc-updown-5m-1777496400",
            start_ts=1777496400,
            end_ts=1777496700,
            settlement_threshold=None,
        )
        embedded_html = """
        <script id="__NEXT_DATA__" type="application/json">
        {"buildId":"build-TfctsWXpff2fKS","pageProps":{"event":{"id":"1"}}}
        </script>
        """
        next_data_payload = {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "state": {
                                "data": {
                                    "data": {
                                        "results": [
                                            {
                                                "startTime": "2026-04-29T20:55:00.000Z",
                                                "endTime": "2026-04-29T21:00:00.000Z",
                                                "openPrice": 75820.0,
                                                "closePrice": 75855.07855,
                                                "outcome": "up",
                                                "percentChange": 0.0462,
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

        with patch(
            "custom.btc_agent.market_lookup._fetch_vatic_price_to_beat_by_slug",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_price_to_beat_by_slug",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_polymarket_page",
            return_value=embedded_html,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_next_data_payload_chain",
            return_value=[("build-TfctsWXpff2fKS", next_data_payload)],
        ):
            hydrated_market = _hydrate_missing_threshold_from_page(
                market,
                "btc-updown-5m-1777496400",
            )

        self.assertEqual(hydrated_market.settlement_threshold, 75855.07855)

    def test_hydrate_missing_threshold_uses_prior_close_only_after_open_sources_fail(self):
        market = BtcUpDownMarket(
            event_id="1",
            market_id="2",
            up_token_id="up-token",
            down_token_id="down-token",
            title="Bitcoin Up or Down - April 29, 5:00PM-5:05PM ET",
            question="Bitcoin Up or Down - April 29, 5:00PM-5:05PM ET",
            slug="btc-updown-5m-1777496400",
            start_ts=1777496400,
            end_ts=1777496700,
            settlement_threshold=None,
        )
        embedded_html = """
        <script id="__NEXT_DATA__" type="application/json">
        {"buildId":"build-TfctsWXpff2fKS","pageProps":{"event":{"id":"1"}}}
        </script>
        """
        next_data_payload = {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "state": {
                                "data": {
                                    "data": {
                                        "results": [
                                            {
                                                "startTime": "2026-04-29T20:55:00.000Z",
                                                "endTime": "2026-04-29T21:00:00.000Z",
                                                "openPrice": 75820.0,
                                                "closePrice": 75855.07855,
                                                "outcome": "up",
                                                "percentChange": 0.0462,
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

        with patch(
            "custom.btc_agent.market_lookup._fetch_vatic_price_to_beat_by_slug",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_price_to_beat_by_slug",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_polymarket_page",
            return_value=embedded_html,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_next_data_payload_chain",
            return_value=[("build-TfctsWXpff2fKS", next_data_payload)],
        ):
            hydrated_market = _hydrate_missing_threshold_from_page(
                market,
                "btc-updown-5m-1777496400",
            )

        self.assertEqual(hydrated_market.settlement_threshold, 75855.07855)

    def test_hydrate_missing_threshold_uses_later_next_data_page_for_prior_close(self):
        market = BtcUpDownMarket(
            event_id="1",
            market_id="2",
            up_token_id="up-token",
            down_token_id="down-token",
            title="Bitcoin Up or Down - April 29, 7:45PM-7:50PM ET",
            question="Bitcoin Up or Down - April 29, 7:45PM-7:50PM ET",
            slug="btc-updown-5m-1777503900",
            start_ts=1777503900,
            end_ts=1777504200,
            settlement_threshold=None,
        )
        embedded_html = """
        <script id="__NEXT_DATA__" type="application/json">
        {"buildId":"build-TfctsWXpff2fKS","pageProps":{"event":{"id":"1"}}}
        </script>
        """
        payload_one = {"pageProps": {"dehydratedState": {"queries": []}}}
        payload_two = {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "state": {
                                "data": {
                                    "data": {
                                        "results": [
                                            {
                                                "startTime": "2026-04-29T23:00:00.000Z",
                                                "endTime": "2026-04-29T23:05:00.000Z",
                                                "openPrice": 75710.0,
                                                "closePrice": 75827.61894026335,
                                                "outcome": "up",
                                                "percentChange": 0.155,
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

        with patch(
            "custom.btc_agent.market_lookup._fetch_vatic_price_to_beat_by_slug",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_price_to_beat_by_slug",
            return_value=None,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_polymarket_page",
            return_value=embedded_html,
        ), patch(
            "custom.btc_agent.market_lookup._fetch_next_data_payload_chain",
            return_value=[("build-TfctsWXpff2fKS", payload_one), ("build-TfctsWXpff2fKS", payload_two)],
        ), patch(
            "custom.btc_agent.market_lookup.os.getcwd",
            return_value="/appl/agents",
        ):
            hydrated_market = _hydrate_missing_threshold_from_page(
                market,
                "btc-updown-5m-1777503900",
            )

        self.assertEqual(hydrated_market.settlement_threshold, 75827.61894026335)

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
            "custom.btc_agent.market_lookup._fetch_polymarket_page",
        ) as mock_fetch_page:
            hydrated_market = _hydrate_missing_threshold_from_page(
                market,
                "btc-updown-5m-1777513800",
            )

        self.assertEqual(hydrated_market.settlement_threshold, 77761.01)
        mock_fetch_page.assert_not_called()

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
        ):
            threshold = _fetch_price_to_beat_by_slug("tesla-up-or-down")

        self.assertEqual(threshold, 77722.39)

    def test_fetch_vatic_price_to_beat_by_slug_parses_price_response(self):
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"price": 77763.01}

        with patch(
            "custom.btc_agent.market_lookup.http_get",
            return_value=mock_response,
        ) as mock_http_get:
            threshold = _fetch_vatic_price_to_beat_by_slug("btc-updown-5m-1777513800")

        self.assertEqual(threshold, 77763.01)
        self.assertEqual(
            mock_http_get.call_args.kwargs["params"],
            {
                "asset": "btc",
                "type": "5min",
                "timestamp": "1777513800",
            },
        )

    def test_fetch_price_to_beat_by_slug_skips_btc_updown_market_slugs(self):
        with patch("custom.btc_agent.market_lookup.http_get") as mock_http_get:
            threshold = _fetch_price_to_beat_by_slug("btc-updown-5m-1776971400")

        self.assertIsNone(threshold)
        mock_http_get.assert_not_called()

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
