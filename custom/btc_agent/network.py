import os
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Optional

import requests


def is_proxy_enabled() -> bool:
    raw_value = os.getenv("USE_PROXY")
    if raw_value is None:
        return True
    return raw_value.strip().lower() in ("1", "true", "yes", "on")


def _get_env_proxy(name: str) -> Optional[str]:
    if not is_proxy_enabled():
        return None
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
    if not is_proxy_enabled():
        return "disabled via USE_PROXY=false"

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


def _should_retry_direct_without_proxy(url: str) -> bool:
    try:
        hostname = (urlsplit(url).hostname or "").lower()
    except Exception:
        return False
    return hostname.endswith("polymarket.com")


def _request_with_direct_timeout_fallback(
    method: str,
    url: str,
    *,
    proxies: Optional[dict],
    request_kwargs: dict,
):
    request_callable = requests.get if method == "GET" else requests.post
    try:
        return request_callable(url, proxies=proxies, **request_kwargs)
    except (requests.ConnectTimeout, requests.ReadTimeout) as exc:
        if not is_proxy_enabled():
            raise
        if proxies is None:
            raise
        if not _should_retry_direct_without_proxy(url):
            raise
        session = requests.Session()
        session.trust_env = False
        try:
            if method == "GET":
                return session.get(url, proxies=None, **request_kwargs)
            return session.post(url, proxies=None, **request_kwargs)
        finally:
            session.close()


def http_get(url: str, **kwargs: Any) -> requests.Response:
    request_kwargs = dict(kwargs)
    proxies = request_kwargs.pop("proxies", None)
    if not is_proxy_enabled():
        session = requests.Session()
        session.trust_env = False
        try:
            return session.get(url, proxies=proxies, **request_kwargs)
        finally:
            session.close()
    if proxies is None:
        proxy_url = get_proxy_url_for_requests("https" if url.startswith("https://") else "http")
        if proxy_url:
            proxies = {"http": proxy_url, "https": proxy_url}

    return _request_with_direct_timeout_fallback(
        "GET",
        url,
        proxies=proxies,
        request_kwargs=request_kwargs,
    )


def http_post(url: str, **kwargs: Any) -> requests.Response:
    request_kwargs = dict(kwargs)
    proxies = request_kwargs.pop("proxies", None)
    if not is_proxy_enabled():
        session = requests.Session()
        session.trust_env = False
        try:
            return session.post(url, proxies=proxies, **request_kwargs)
        finally:
            session.close()
    if proxies is None:
        proxy_url = get_proxy_url_for_requests("https" if url.startswith("https://") else "http")
        if proxy_url:
            proxies = {"http": proxy_url, "https": proxy_url}

    return _request_with_direct_timeout_fallback(
        "POST",
        url,
        proxies=proxies,
        request_kwargs=request_kwargs,
    )


def check_internet_connectivity(timeout: float = 5.0) -> tuple[bool, str]:
    test_url = "https://www.google.com/generate_204"
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(test_url, timeout=timeout)
        response.raise_for_status()
        return True, f"Connectivity OK via {test_url} (HTTP {response.status_code})"
    except requests.RequestException as exc:
        return False, f"Connectivity check failed via {test_url}: {exc}"
    finally:
        session.close()
