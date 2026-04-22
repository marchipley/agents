import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))

from scripts.python.check_public_ip_indonesia import check_current_public_ip_location


class TestCheckPublicIpFailure(unittest.TestCase):
    def test_public_ip_failure_surfaces_message(self):
        with patch(
            "scripts.python.check_public_ip_indonesia.get_public_ip",
            return_value=(None, "Unable to determine public IP address: proxy unreachable"),
        ):
            public_ip, location, is_allowed = check_current_public_ip_location()

        self.assertIsNone(public_ip)
        self.assertFalse(is_allowed)
        self.assertEqual(
            location["message"],
            "Unable to determine public IP address: proxy unreachable",
        )


if __name__ == "__main__":
    unittest.main()
