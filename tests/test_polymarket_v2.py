import unittest
import logging
from unittest.mock import Mock, patch

from agents.polymarket.polymarket import Polymarket, _configure_v2_sdk_logging


class TestPolymarketV2(unittest.TestCase):
    def test_configure_v2_sdk_logging_suppresses_sdk_loggers(self):
        logger_names = (
            "py_clob_client_v2",
            "py_clob_client_v2.client",
            "py_clob_client_v2.http_helpers",
            "py_clob_client_v2.http_helpers.helpers",
        )

        try:
            for logger_name in logger_names:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.NOTSET)
                logger.propagate = True

            _configure_v2_sdk_logging()

            for logger_name in logger_names:
                logger = logging.getLogger(logger_name)
                self.assertEqual(logger.level, logging.CRITICAL)
                self.assertFalse(logger.propagate)
        finally:
            for logger_name in logger_names:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.NOTSET)
                logger.propagate = True

    def test_init_uses_v2_client_and_derives_api_key(self):
        bootstrap_client = Mock()
        bootstrap_client.create_or_derive_api_key.return_value = "creds"
        authed_client = Mock()
        mock_sdk = {
            "ClobClient": Mock(side_effect=[bootstrap_client, authed_client]),
            "MarketOrderArgs": Mock(),
            "OrderArgs": Mock(),
            "OrderType": Mock(GTC="GTC", FOK="FOK"),
            "PartialCreateOrderOptions": Mock(),
            "Side": Mock(BUY="BUY", SELL="SELL"),
        }

        with patch(
            "agents.polymarket.polymarket._load_v2_sdk",
            return_value=mock_sdk,
        ), patch.object(
            Polymarket,
            "_init_approvals",
        ):
            poly = Polymarket()

        self.assertEqual(poly.credentials, "creds")
        bootstrap_client.create_or_derive_api_key.assert_called_once_with()
        self.assertIs(poly.client, authed_client)
        self.assertEqual(mock_sdk["ClobClient"].call_count, 2)

        first_kwargs = mock_sdk["ClobClient"].call_args_list[0].kwargs
        second_kwargs = mock_sdk["ClobClient"].call_args_list[1].kwargs
        self.assertEqual(first_kwargs["host"], "https://clob.polymarket.com")
        self.assertEqual(first_kwargs["chain_id"], 137)
        self.assertEqual(second_kwargs["creds"], "creds")

    def test_execute_order_posts_v2_limit_order_with_tick_size(self):
        order_args_cls = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
        options_cls = Mock(side_effect=lambda **kwargs: Mock(**kwargs))

        poly = Polymarket.__new__(Polymarket)
        poly.client = Mock()
        poly._OrderArgs = order_args_cls
        poly._PartialCreateOrderOptions = options_cls
        poly._OrderType = Mock(GTC="GTC")

        with patch(
            "agents.polymarket.polymarket._load_v2_sdk",
            return_value={"Side": Mock(BUY="BUY", SELL="SELL")},
        ):
            poly.execute_order(
                price=0.421,
                size=5.0,
                side="BUY",
                token_id="token-1",
                fee_rate_bps=1000,
                tick_size=0.001,
            )

        kwargs = poly.client.create_and_post_order.call_args.kwargs
        order_args = kwargs["order_args"]
        options = kwargs["options"]

        self.assertEqual(order_args.token_id, "token-1")
        self.assertEqual(order_args.price, 0.421)
        self.assertEqual(order_args.size, 5.0)
        self.assertEqual(order_args.side, "BUY")
        self.assertEqual(options.tick_size, "0.001")
        self.assertEqual(kwargs["order_type"], "GTC")


if __name__ == "__main__":
    unittest.main()
