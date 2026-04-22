import os
import unittest
from unittest.mock import patch

from custom.btc_agent.network import (
    describe_proxy_configuration,
    get_proxy_url_for_httpx,
    get_proxy_url_for_requests,
    mask_proxy_url,
)


class TestBtcNetwork(unittest.TestCase):
    def test_requests_prefers_all_proxy_and_upgrades_socks5_to_socks5h(self):
        with patch.dict(
            os.environ,
            {"ALL_PROXY": "socks5://10.64.0.1:1080"},
            clear=False,
        ):
            self.assertEqual(
                get_proxy_url_for_requests("https"),
                "socks5h://10.64.0.1:1080",
            )

    def test_httpx_prefers_all_proxy_and_uses_socks5_scheme(self):
        with patch.dict(
            os.environ,
            {"ALL_PROXY": "socks5h://10.64.0.1:1080"},
            clear=False,
        ):
            self.assertEqual(
                get_proxy_url_for_httpx(),
                "socks5://10.64.0.1:1080",
            )

    def test_requests_falls_back_to_https_then_http_proxy(self):
        with patch.dict(
            os.environ,
            {
                "ALL_PROXY": "",
                "HTTPS_PROXY": "http://proxy.example:443",
                "HTTP_PROXY": "http://proxy.example:80",
            },
            clear=False,
        ):
            self.assertEqual(
                get_proxy_url_for_requests("https"),
                "http://proxy.example:443",
            )
            self.assertEqual(
                get_proxy_url_for_requests("http"),
                "http://proxy.example:80",
            )

    def test_mask_proxy_url_hides_password(self):
        self.assertEqual(
            mask_proxy_url("socks5h://user:secret@10.64.0.1:1080"),
            "socks5h://user:***@10.64.0.1:1080",
        )

    def test_describe_proxy_configuration_reports_active_env(self):
        with patch.dict(
            os.environ,
            {"ALL_PROXY": "socks5h://user:secret@10.64.0.1:1080"},
            clear=False,
        ):
            self.assertEqual(
                describe_proxy_configuration(),
                "enabled via ALL_PROXY=socks5h://user:***@10.64.0.1:1080",
            )


if __name__ == "__main__":
    unittest.main()
