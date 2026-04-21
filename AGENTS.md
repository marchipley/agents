# AGENTS.md

## Project Identity

This repository is a fork of `polymarket/agents` that is being adapted into a focused BTC Up/Down trading agent for Polymarket on Polygon.

Current intent:

- Analyze the active 5-minute BTC Up/Down market.
- Build a lightweight BTC feature snapshot.
- Ask an OpenAI model for a directional decision.
- Simulate execution with paper trades only.
- Add live trading only after paper-trading behavior is validated.

The active custom implementation lives under `custom/btc_agent/`. The inherited `agents/` tree remains available, but most of it is upstream framework code and is not the main runtime for the BTC-specific workflow.

## Current Runtime Path

Primary launcher:

- [launch_btc_agent.sh](/appl/agents/launch_btc_agent.sh:1)

Primary module:

- [custom/btc_agent/main.py](/appl/agents/custom/btc_agent/main.py:1)

Execution flow per loop tick:

1. Load env/config from [custom/btc_agent/config.py](/appl/agents/custom/btc_agent/config.py:1).
2. Run the public-IP geolocation check from [scripts/python/check_public_ip_indonesia.py](/appl/agents/scripts/python/check_public_ip_indonesia.py:1).
3. Abort startup immediately if the public IP does not resolve to Indonesia.
4. Print wallet/account balances via helpers in [custom/btc_agent/executor.py](/appl/agents/custom/btc_agent/executor.py:1).
5. Resolve the active BTC Up/Down market slug in [custom/btc_agent/market_lookup.py](/appl/agents/custom/btc_agent/market_lookup.py:1).
6. Fetch current quotes for both outcome tokens.
7. Build BTC features in [custom/btc_agent/indicators.py](/appl/agents/custom/btc_agent/indicators.py:1).
8. Request an LLM trade decision in [custom/btc_agent/llm_decision.py](/appl/agents/custom/btc_agent/llm_decision.py:1).
9. Reprice the selected token and evaluate whether a paper trade would be allowed.
10. Print the paper-trade decision and sleep for the configured interval.

There is currently no live order submission in the custom BTC loop.

## Repository Map

Custom BTC agent:

- `custom/btc_agent/config.py`: env loading and typed config objects.
- `custom/btc_agent/main.py`: long-running loop and stdout logging.
- `custom/btc_agent/market_lookup.py`: event slug selection and token ID extraction.
- `custom/btc_agent/indicators.py`: BTC spot fetch plus simple in-memory indicators.
- `custom/btc_agent/llm_decision.py`: OpenAI prompt/response handling.
- `custom/btc_agent/executor.py`: account snapshots, quote inspection, limit-price logic, paper-trade gatekeeping.

Inherited upstream framework:

- `agents/polymarket/`: general Polymarket and Gamma clients, including live-trading primitives from the upstream framework.
- `agents/application/`: generic LLM/RAG trading flow from the original project.
- `agents/connectors/`: news/search/chroma integrations.
- `agents/utils/objects.py`: shared Pydantic models.

Scripts:

- `launch_btc_agent.sh`: local launcher for the BTC agent.
- `scripts/python/check_public_ip_indonesia.py`: standalone utility to print the public IP and geolocation, then return success only when the geolocation resolves to Indonesia.
- `scripts/python/cli.py`: inherited Typer CLI for the upstream agent workflow.

Tests:

- `tests/test.py` is only placeholder unittest content from the fork base and does not validate BTC-agent behavior.

## Environment And Configuration

Required for current BTC agent execution:

- `POLYGON_WALLET_PRIVATE_KEY`
- `OPENAI_API_KEY`

Optional / supported:

- `OPENAI_MODEL` default: `gpt-4.1-mini`
- `POLYMKT_PROXY_ADDRESS`
- `POLYGON_RPC_URL` default: `https://polygon.drpc.org`
- `POLYGON_RPC_URLS` optional comma-separated list of Polygon RPC endpoints to try in order
- `BTC_AGENT_LOOP_INTERVAL` default: `30`
- `BTC_AGENT_MAX_TRADE_USD` default: `5`
- `BTC_AGENT_MIN_CONFIDENCE` default: `0.7`
- `BTC_AGENT_MAX_ENTRY_PRICE` default: `0.62`
- `BTC_AGENT_MAX_SPREAD` default: `0.06`
- `BTC_AGENT_MARKET_SLUG` for override/debugging/backtesting

Notes:

- `.env.example` does not yet document the BTC-agent-specific variables and should be updated when configuration is stabilized.
- `config.py` loads `.env` from the repo root, so local execution assumes a root-level `.env`.

## Current Behavior

What the BTC agent does today:

- Pulls the current BTC/USD spot price from CoinGecko.
- Maintains an in-memory rolling price history during process lifetime only.
- Approximates window-open price using the earliest retained sample, not a true historical open for the market window.
- Uses the current 5-minute BTC Up/Down slug by timestamp alignment, unless overridden.
- Performs a startup IP geolocation check and refuses to run unless the current public IP resolves to Indonesia.
- Uses OpenAI chat completions with JSON output to decide `UP`, `DOWN`, or `NO_TRADE`.
- Computes a reference price from quote, midpoint, last trade, and order book data.
- Retrieves Polygon USDC cash balances through a configurable ordered RPC list with public fallback endpoints.
- Retrieves Polymarket portfolio value separately from the on-chain cash balance lookup so one failure does not suppress the other.
- Approves or rejects a paper trade based on confidence, entry caps, and quote drift.
- Prints diagnostics for balances, quotes, features, decision, and simulated execution.

What it does not do yet:

- Persist price history or trading state across restarts.
- Use a true market-window open from historical BTC data.
- Submit live Polymarket orders from the custom BTC flow.
- Record trades, decisions, or metrics to disk or a database.
- Contain meaningful automated tests for BTC-specific logic.

## Important Inherited Code Context

The repository still contains upstream code for broader autonomous market selection and execution:

- [agents/application/trade.py](/appl/agents/agents/application/trade.py:1) runs the old generic workflow.
- [agents/application/executor.py](/appl/agents/agents/application/executor.py:1) contains generic prompt-heavy logic using LangChain and Chroma.
- [agents/polymarket/polymarket.py](/appl/agents/agents/polymarket/polymarket.py:1) includes lower-level market and execution capabilities that will likely be reused when live BTC trading is implemented.

When changing live-trading behavior, prefer reusing vetted primitives from `agents/polymarket/` rather than rebuilding order plumbing from scratch, but keep the BTC strategy logic isolated under `custom/btc_agent/`.

## Known Gaps And Risks

- `BTC_AGENT_MAX_SPREAD` exists in config but is not currently enforced in `custom/btc_agent/executor.py`.
- The indicator pipeline depends on process-local memory, so restarts erase context and make RSI/momentum less meaningful until enough samples accumulate.
- The market lookup assumes the current slug format and first market in the event response remain stable.
- The decision path depends on a single LLM response and does not yet validate response quality beyond JSON parsing and basic coercion.
- Network/API failures are still only partially hardened; balance lookups now degrade more cleanly, but other external calls remain single-point dependent.
- There are no BTC-agent tests for market parsing, pricing, decision normalization, or paper-trade gating.
- The inherited repo still contains placeholder tests and unused upstream surfaces, which can mislead future work if not distinguished from the active BTC path.

## Ongoing Development Guidance

When modifying this repo, future Codex runs should:

- Treat `custom/btc_agent/` as the primary application area unless the task is explicitly about upstream framework code.
- Preserve paper-trading-only behavior unless the user explicitly requests live trading changes.
- Avoid changing unrelated upstream modules just because they exist; keep BTC-specific logic isolated where practical.
- Add or update tests when changing pricing, market selection, decision parsing, or execution gating.
- Update this file whenever architecture, runtime flow, env vars, or user-facing behavior changes.

If live trading is introduced later, this file should be updated to document:

- Which module actually submits orders.
- Which approvals/allowances are required.
- Whether proxy wallets are supported in that path.
- The exact safeguards that gate live execution.

## Recent Project History

Known repo history from git:

- Upstream-derived BTC branch baseline: commit `6260f4e`
  - message: `Polymarket agent using AI to detect signals and execute trades for BTC UP/Down`
- Current branch head: commit `e7f7e76`
  - message: `Polymarket does not allow certain locations so we need to ensure the IP address resides in a non restricted location`

Current local work includes:

- `scripts/python/check_public_ip_indonesia.py`
- `AGENTS.md`

Do not revert unrelated local changes unless the user explicitly asks for that.

## Change Log

### 2026-04-21

- Added `AGENTS.md` to document the fork’s active BTC-agent architecture, configuration surface, inherited vs active code paths, and maintenance rules for future Codex work.
- Added `scripts/python/check_public_ip_indonesia.py` to print the current public IP and geolocation and return success only when the detected location is Indonesia.
- Recreated `AGENTS.md` after accidental deletion, preserving the prior project context and maintenance guidance.
- Updated BTC account balance handling to stop using `https://polygon-rpc.com` as the default Polygon RPC, add ordered RPC fallback support via `POLYGON_RPC_URLS`, and separate cash-balance failures from portfolio-balance failures in the printed account snapshot.
- Added a hard startup gate in `custom/btc_agent/main.py` that runs the Indonesia public-IP check and aborts execution before any further BTC-agent logic when the detected location is not Indonesia.
- Adjusted `scripts/python/check_public_ip_indonesia.py` type annotations to remain compatible with the repo’s Python 3.9 runtime after the Indonesia startup gate was added.
