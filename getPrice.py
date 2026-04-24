import argparse
import re
import sys
from decimal import Decimal, InvalidOperation

import requests
from bs4 import BeautifulSoup


def extract_value_from_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    target_classes = {"text-text-secondary", "text-heading-2xl"}

    for span in soup.find_all("span"):
        classes = set(span.get("class", []))
        if target_classes.issubset(classes):
            text = span.get_text(" ", strip=True)
            match = re.search(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)", text)
            if match:
                numeric_text = match.group(1).replace(",", "")
                try:
                    return str(Decimal(numeric_text))
                except InvalidOperation:
                    pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a URL, find a span containing classes 'text-text-secondary' and 'text-heading-2xl', and print the numeric value."
    )
    parser.add_argument("url", help="Website URL to inspect")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout in seconds (default: 20)")
    args = parser.parse_args()

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }

    try:
        response = requests.get(args.url, headers=headers, timeout=args.timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    value = extract_value_from_html(response.text)
    if value is None:
        print("No matching span/value found.", file=sys.stderr)
        sys.exit(2)

    print(value)


if __name__ == "__main__":
    main()