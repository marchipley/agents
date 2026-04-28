import os
import unittest
from unittest.mock import MagicMock, patch

import requests

from custom.btc_agent.network import (
    check_internet_connectivity,
    describe_proxy_configuration,
    http_post,
    get_proxy_url_for_httpx,
    get_proxy_url_for_requests,
    is_proxy_enabled,
    mask_proxy_url,
)


class TestBtcNetwork(unittest.TestCase):
    def test_proxy_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(is_proxy_enabled())

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

    def test_use_proxy_false_disables_proxy_resolution(self):
        with patch.dict(
            os.environ,
            {
                "USE_PROXY": "false",
                "ALL_PROXY": "socks5h://10.64.0.1:1080",
                "HTTPS_PROXY": "http://proxy.example:443",
                "HTTP_PROXY": "http://proxy.example:80",
            },
            clear=False,
        ):
            self.assertFalse(is_proxy_enabled())
            self.assertIsNone(get_proxy_url_for_requests("https"))
            self.assertIsNone(get_proxy_url_for_httpx())
            self.assertEqual(
                describe_proxy_configuration(),
                "disabled via USE_PROXY=false",
            )

    def test_http_post_disables_trust_env_when_use_proxy_false(self):
        fake_session = MagicMock()
        fake_response = object()
        fake_session.post.return_value = fake_response

        with patch.dict(
            os.environ,
            {
                "USE_PROXY": "false",
                "ALL_PROXY": "socks5h://10.64.0.1:1080",
            },
            clear=False,
        ), patch(
            "custom.btc_agent.network.requests.Session",
            return_value=fake_session,
        ):
            response = http_post("https://example.com", json={"ok": True}, timeout=5)

        self.assertIs(response, fake_response)
        self.assertFalse(fake_session.trust_env)
        fake_session.post.assert_called_once()
        fake_session.close.assert_called_once()

    def test_check_internet_connectivity_reports_success(self):
        fake_session = MagicMock()
        response = MagicMock()
        response.status_code = 204
        fake_session.get.return_value = response

        with patch(
            "custom.btc_agent.network.requests.Session",
            return_value=fake_session,
        ):
            ok, detail = check_internet_connectivity(timeout=3.0)

        self.assertTrue(ok)
        self.assertIn("Connectivity OK", detail)
        self.assertIn("HTTP 204", detail)
        self.assertFalse(fake_session.trust_env)
        fake_session.close.assert_called_once()

    def test_check_internet_connectivity_reports_failure(self):
        fake_session = MagicMock()
        fake_session.get.side_effect = requests.RequestException("boom")

        with patch(
            "custom.btc_agent.network.requests.Session",
            return_value=fake_session,
        ):
            ok, detail = check_internet_connectivity(timeout=3.0)

        self.assertFalse(ok)
        self.assertIn("Connectivity check failed", detail)
        self.assertFalse(fake_session.trust_env)
        fake_session.close.assert_called_once()

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
