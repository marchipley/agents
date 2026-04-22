import os
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Optional

import requests


def _get_env_proxy(name: str) -> Optional[str]:
    value = os.getenv(name) or os.getenv(name.lower())
    if not value:
        return None
    return value.strip() or None


def _normalize_requests_proxy_url(proxy_url: str) -> str:
    if proxy_url.startswith("socks5://"):
        return "socks5h://" + proxy_url[len("socks5://") :]
    return proxy_url


def _normalize_httpx_proxy_url(proxy_url: str) -> str:
    if proxy_url.startswith("socks5h://"):
        return "socks5://" + proxy_url[len("socks5h://") :]
    return proxy_url


def get_proxy_url_for_requests(scheme: str) -> Optional[str]:
    all_proxy = _get_env_proxy("ALL_PROXY")
    if all_proxy:
        return _normalize_requests_proxy_url(all_proxy)

    if scheme == "https":
        proxy = _get_env_proxy("HTTPS_PROXY")
        if proxy:
            return _normalize_requests_proxy_url(proxy)

    proxy = _get_env_proxy("HTTP_PROXY")
    if proxy:
        return _normalize_requests_proxy_url(proxy)

    return None


def get_proxy_url_for_httpx() -> Optional[str]:
    all_proxy = _get_env_proxy("ALL_PROXY")
    if all_proxy:
        return _normalize_httpx_proxy_url(all_proxy)

    https_proxy = _get_env_proxy("HTTPS_PROXY")
    if https_proxy:
        return _normalize_httpx_proxy_url(https_proxy)

    http_proxy = _get_env_proxy("HTTP_PROXY")
    if http_proxy:
        return _normalize_httpx_proxy_url(http_proxy)

    return None


def mask_proxy_url(proxy_url: Optional[str]) -> str:
    if not proxy_url:
        return "None"

    try:
        parts = urlsplit(proxy_url)
    except Exception:
        return proxy_url

    hostname = parts.hostname or ""
    if not hostname:
        return proxy_url

    auth_prefix = ""
    if parts.username is not None:
        auth_prefix = parts.username
        if parts.password is not None:
            auth_prefix += ":***"
        auth_prefix += "@"

    netloc = auth_prefix + hostname
    if parts.port is not None:
        netloc += f":{parts.port}"

    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def describe_proxy_configuration() -> str:
    all_proxy = _get_env_proxy("ALL_PROXY")
    https_proxy = _get_env_proxy("HTTPS_PROXY")
    http_proxy = _get_env_proxy("HTTP_PROXY")

    if all_proxy:
        return f"enabled via ALL_PROXY={mask_proxy_url(all_proxy)}"
    if https_proxy:
        return f"enabled via HTTPS_PROXY={mask_proxy_url(https_proxy)}"
    if http_proxy:
        return f"enabled via HTTP_PROXY={mask_proxy_url(http_proxy)}"
    return "disabled"


def http_get(url: str, **kwargs: Any) -> requests.Response:
    request_kwargs = dict(kwargs)
    proxies = request_kwargs.pop("proxies", None)
    if proxies is None:
        proxy_url = get_proxy_url_for_requests("https" if url.startswith("https://") else "http")
        if proxy_url:
            proxies = {"http": proxy_url, "https": proxy_url}

    return requests.get(url, proxies=proxies, **request_kwargs)


def http_post(url: str, **kwargs: Any) -> requests.Response:
    request_kwargs = dict(kwargs)
    proxies = request_kwargs.pop("proxies", None)
    if proxies is None:
        proxy_url = get_proxy_url_for_requests("https" if url.startswith("https://") else "http")
        if proxy_url:
            proxies = {"http": proxy_url, "https": proxy_url}

    return requests.post(url, proxies=proxies, **request_kwargs)
