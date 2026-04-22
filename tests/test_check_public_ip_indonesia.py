import unittest

from scripts.python.check_public_ip_indonesia import is_allowed_location


class TestCheckPublicIpIndonesia(unittest.TestCase):
    def test_indonesia_is_allowed(self):
        self.assertTrue(is_allowed_location({"country": "Indonesia"}))
        self.assertTrue(is_allowed_location({"country_code": "ID"}))

    def test_mexico_is_allowed(self):
        self.assertTrue(is_allowed_location({"country": "Mexico"}))
        self.assertTrue(is_allowed_location({"country_code": "MX"}))

    def test_other_country_is_not_allowed(self):
        self.assertFalse(is_allowed_location({"country": "United States"}))
        self.assertFalse(is_allowed_location({"country_code": "US"}))


if __name__ == "__main__":
    unittest.main()
