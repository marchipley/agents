import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))

from custom.btc_agent.market_lookup import (
    build_price_to_beat_debug_report,
    _extract_next_build_id,
    _extract_event_from_next_data,
    _extract_live_period_open_from_next_data,
    _extract_market_from_event,
    _extract_previous_period_close_from_next_data,
    _extract_previous_period_final_price_from_next_data,
    _extract_threshold_from_price_to_beat_response,
    _fetch_event_from_next_data_route,
    _fetch_next_data_payload,
    _fetch_price_to_beat_by_slug,
    _extract_threshold_from_page_html,
    _parse_threshold_from_text,
)


class TestBtcMarketLookup(unittest.TestCase):
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

    def test_extract_previous_period_final_price_from_next_data_uses_matching_end_date(self):
        payload = {
            "pageProps": {
                "events": [
                    {
                        "slug": "btc-updown-5m-1777055700",
                        "endDate": "2026-04-24T18:40:00Z",
                        "eventMetadata": {
                            "finalPrice": 77560.75,
                            "priceToBeat": 77519.716,
                        },
                    },
                    {
                        "slug": "btc-updown-5m-1777056000",
                        "endDate": "2026-04-24T18:45:00Z",
                        "eventMetadata": {
                            "finalPrice": 77598.79949998436,
                            "priceToBeat": 77560.75,
                        },
                    },
                ]
            }
        }

        threshold = _extract_previous_period_final_price_from_next_data(
            payload,
            "btc-updown-5m-1777056000",
        )

        self.assertEqual(threshold, 77560.75)

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
                },
                "events": [
                    {
                        "slug": "btc-updown-5m-1776982800",
                        "endDate": "2026-04-23T22:25:00Z",
                        "eventMetadata": {
                            "finalPrice": 78210.0,
                        },
                    }
                ],
            }
        }

        with patch(
            "custom.btc_agent.market_lookup._fetch_polymarket_page",
            return_value='<script id="__NEXT_DATA__" type="application/json">{"buildId":"build-TfctsWXpff2fKS"}</script>',
        ), patch(
            "custom.btc_agent.market_lookup._fetch_next_data_payload",
            return_value=payload,
        ):
            report = build_price_to_beat_debug_report("btc-updown-5m-1776983100")

        self.assertIn("next_data_curl=curl 'https://polymarket.com/_next/data/build-TfctsWXpff2fKS/en/event/btc-updown-5m-1776983100.json?slug=btc-updown-5m-1776983100'", report)
        self.assertIn("live_period_open=78218.01972274295", report)
        self.assertIn("previous_period_final_price_from_event_metadata=78210.0", report)

    def test_extract_threshold_from_price_to_beat_response_handles_nested_payload(self):
        threshold = _extract_threshold_from_price_to_beat_response(
            {
                "data": {
                    "priceToBeat": "77722.39",
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
