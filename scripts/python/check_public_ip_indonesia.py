#!/usr/bin/env python3

import argparse
import sys
from typing import Any

import requests


IPIFY_URL = "https://api.ipify.org?format=json"
IPWHOIS_URL_TEMPLATE = "https://ipwho.is/{ip}"


def get_public_ip(timeout: float) -> str | None:
    try:
        response = requests.get(IPIFY_URL, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None

    ip = payload.get("ip")
    return ip if isinstance(ip, str) and ip else None


def get_ip_location(ip: str, timeout: float) -> dict[str, Any]:
    try:
        response = requests.get(IPWHOIS_URL_TEMPLATE.format(ip=ip), timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {
            "success": False,
            "message": f"Lookup failed: {exc}",
            "ip": ip,
        }

    if "ip" not in payload:
        payload["ip"] = ip

    return payload


def is_indonesia(location: dict[str, Any]) -> bool:
    country = str(location.get("country", "")).strip().lower()
    country_code = str(location.get("country_code", "")).strip().lower()
    continent = str(location.get("continent", "")).strip().lower()

    return (
        country == "indonesia"
        or country_code == "id"
        or (country == "indonesia" and continent == "asia")
    )


def print_location(ip: str | None, location: dict[str, Any]) -> None:
    print(f"public_ip: {ip or 'unknown'}")
    print(f"is_indonesia: {str(is_indonesia(location)).lower()}")
    print(f"lookup_success: {str(bool(location.get('success', False))).lower()}")
    print(f"country: {location.get('country', 'unknown')}")
    print(f"country_code: {location.get('country_code', 'unknown')}")
    print(f"region: {location.get('region', 'unknown')}")
    print(f"city: {location.get('city', 'unknown')}")
    print(f"continent: {location.get('continent', 'unknown')}")
    print(f"latitude: {location.get('latitude', 'unknown')}")
    print(f"longitude: {location.get('longitude', 'unknown')}")
    print(f"asn: {location.get('connection', {}).get('asn', 'unknown')}")
    print(f"org: {location.get('connection', {}).get('org', 'unknown')}")

    message = location.get("message")
    if message:
        print(f"message: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the current public IP and its geolocation, then exit with "
            "status 0 if the IP is in Indonesia and 1 otherwise."
        )
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds for each lookup request.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    public_ip = get_public_ip(timeout=args.timeout)

    if public_ip is None:
        location = {
            "success": False,
            "message": "Unable to determine public IP address.",
        }
        print_location(None, location)
        return 1

    location = get_ip_location(public_ip, timeout=args.timeout)
    print_location(public_ip, location)
    return 0 if is_indonesia(location) else 1


if __name__ == "__main__":
    sys.exit(main())
