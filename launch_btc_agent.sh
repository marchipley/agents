#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

WINDOWS_HOST_IP="${LLM_WINDOWS_HOST_IP:-$(grep nameserver /etc/resolv.conf 2>/dev/null | awk 'NR==1 {print $2; exit}')}"
if [[ -z "${WINDOWS_HOST_IP:-}" ]]; then
  WINDOWS_HOST_IP="10.255.255.254"
fi
VPN_PROXY_URL="${VPN_PROXY_URL:-socks5h://10.64.0.1:1080}"
LLM_PROXY_PORT="${LLM_PROXY_PORT:-8888}"
export ALL_PROXY="$VPN_PROXY_URL"
export all_proxy="$VPN_PROXY_URL"
export HTTP_PROXY="$VPN_PROXY_URL"
export http_proxy="$VPN_PROXY_URL"
export HTTPS_PROXY="$VPN_PROXY_URL"
export https_proxy="$VPN_PROXY_URL"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="$NO_PROXY"
export LLM_WINDOWS_HOST_IP="$WINDOWS_HOST_IP"
export LLM_PROXY_PORT="$LLM_PROXY_PORT"
export LLM_PROXY_URL="${LLM_PROXY_URL:-http://${WINDOWS_HOST_IP}:${LLM_PROXY_PORT}}"

if [[ "${VIRTUAL_ENV:-}" != "$REPO_ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="."

python -m custom.btc_agent.main
