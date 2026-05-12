# AGENTS.md

## Project Identity

This repository is a fork of `polymarket/agents` that is being adapted into a focused BTC Up/Down trading agent for Polymarket on Polygon.

Current intent:

- Analyze the active 5-minute BTC Up/Down market.
- Build a lightweight BTC feature snapshot.
- Ask the configured AI engine for a directional decision.
- Support paper trading by default and optional live order submission when explicitly enabled.

The active custom implementation lives under `custom/btc_agent/`. The inherited `agents/` tree remains available, but most of it is upstream framework code and is not the main runtime for the BTC-specific workflow.

## Current Runtime Path

Primary launcher:

- [launch_btc_agent.sh](/appl/agents/launch_btc_agent.sh:1)

Primary module:

- [custom/btc_agent/main.py](/appl/agents/custom/btc_agent/main.py:1)

Execution flow per loop tick:

1. Load env/config from [custom/btc_agent/config.py](/appl/agents/custom/btc_agent/config.py:1).
2. Run the public-IP geolocation check from [scripts/python/check_public_ip_indonesia.py](/appl/agents/scripts/python/check_public_ip_indonesia.py:1).
3. Abort startup immediately if the public IP does not resolve to an allowed location, currently Indonesia or Mexico.
4. Print wallet/account balances via helpers in [custom/btc_agent/executor.py](/appl/agents/custom/btc_agent/executor.py:1).
5. Resolve the active BTC Up/Down market slug in [custom/btc_agent/market_lookup.py](/appl/agents/custom/btc_agent/market_lookup.py:1).
6. Fetch current quotes for both outcome tokens.
7. Build BTC features in [custom/btc_agent/indicators.py](/appl/agents/custom/btc_agent/indicators.py:1).
8. Request an LLM trade decision in [custom/btc_agent/llm_decision.py](/appl/agents/custom/btc_agent/llm_decision.py:1).
9. If the configured per-period paper-trade limit has already been reached for the current 5-minute market slug, skip quote snapshots and LLM decisioning, print active paper-order status only, and wait for the next loop tick.
10. Otherwise, fetch a decision-time quote snapshot for the selected token, print it, and reuse that same snapshot for paper-trade evaluation within the tick.
11. If a trade executes, record an in-memory active order for the current market window using the configured share size.
12. In non-debug mode, print only compact operational output for geolocation, balances, quotes, features, the LLM decision, and the final execution result; in debug mode, print the fuller diagnostic output.
13. Execute a paper trade when `USE_PAPER_TRADES=true`, or submit a live Polymarket buy order through the upstream client when `USE_PAPER_TRADES=false`.
14. For live mode, abort the process immediately if the required live trade cash, including the estimated taker-fee buffer, exceeds the available `cash_balance_pusd`, or if the flow cannot safely support the configured wallet path.
15. Print the execution snapshot, execution result, and any active order status, then sleep for the configured interval.

## Repository Map

Custom BTC agent:

- `custom/btc_agent/config.py`: env loading and typed config objects.
- `custom/btc_agent/main.py`: long-running loop and stdout logging.
- `custom/btc_agent/market_lookup.py`: event slug selection and token ID extraction.
- `custom/btc_agent/indicators.py`: BTC spot fetch plus simple in-memory indicators.
- `custom/btc_agent/llm_decision.py`: AI-engine prompt/response handling.
- `custom/btc_agent/executor.py`: account snapshots, quote inspection, limit-price logic, paper-trade gatekeeping.

Inherited upstream framework:

- `agents/polymarket/`: general Polymarket and Gamma clients, including live-trading primitives from the upstream framework.
- `agents/application/`: generic LLM/RAG trading flow from the original project.
- `agents/connectors/`: news/search/chroma integrations.
- `agents/utils/objects.py`: shared Pydantic models.

Scripts:

- `launch_btc_agent.sh`: local launcher for the BTC agent.
- `scripts/python/check_public_ip_indonesia.py`: standalone utility to print the public IP and geolocation, then return success only when the geolocation resolves to an allowed country, currently Indonesia or Mexico.
- `scripts/python/cli.py`: inherited Typer CLI for the upstream agent workflow.

Tests:

- `tests/test.py` is only placeholder unittest content from the fork base and does not validate BTC-agent behavior.

## Environment And Configuration

Required for current BTC agent execution:

- `POLYGON_WALLET_PRIVATE_KEY`
- `AI_ENGINE`
- `OPENAI_API_KEY` when `AI_ENGINE=OPENAI`
- `GEMINI_API_KEY` when `AI_ENGINE=GEMINI`

Optional / supported:

- `AI_ENGINE` supported values: `OPENAI`, `GEMINI`
- `OPENAI_MODEL` default: `gpt-4.1-mini`
- `GEMINI_MODEL` default: `gemini-2.5-flash`
- `API_CONNECTION_TIMEOUT` default: `10`
- `API_CONNECTION_RETRY_TIMER` default: `2.0`
- `API_CONNECTION_RETRY_ATTEMPTS` default: `3`
- `ALL_PROXY` optional global proxy setting for outbound API calls; for Mullvad WireGuard the intended SOCKS5 value is `socks5h://10.64.0.1:1080`
- `HTTP_PROXY` / `HTTPS_PROXY` optional proxy settings for outbound API calls, including LLM requests and geolocation checks
- `NO_PROXY` optional bypass list for local addresses
- `USE_PROXY` default: `true`; when set to `false`, the BTC agent ignores `ALL_PROXY`, `HTTP_PROXY`, and `HTTPS_PROXY` for its shared HTTP/LLM network path
- `LLM_CONNECTION_DEBUG` is controlled by `custom/btc_agent/config.py`; when set to `true`, the agent skips geolocation and the normal trading startup flow, then runs only LLM connectivity diagnostics
- `USE_PAPER_TRADES` is controlled by `custom/btc_agent/config.py`; it currently controls whether the bot paper-trades or submits live Polymarket orders
- `MINIMUM_WALLET_BALANCE` default: `0`; enforced against Polygon pUSD trading cash, not legacy USDC.e
- `BTC_AGENT_LIVE_FEE_RATE_BPS` default: `1000`
- `BTC_AGENT_LIVE_MIN_ORDER_USD` default: `1`
- `USE_RECOMMENDED_LIMIT` default: `true`; when `false`, the BTC agent uses the target limit instead of the recommended/snapped limit for submit checks and execution pricing
- `POLYMKT_PROXY_ADDRESS`
- `POLYGON_RPC_URL` default: `https://polygon.drpc.org`
- `POLYGON_RPC_URLS` optional comma-separated list of Polygon RPC endpoints to try in order
- `BTC_AGENT_DEBUG` is controlled by `custom/btc_agent/config.py`; when `true`, the agent emits fuller diagnostics and forces paper-style debugging behavior
- `BTC_AGENT_DEBUG_WARMUP` is controlled by `custom/btc_agent/config.py`; when `false` and `BTC_AGENT_DEBUG=true`, the agent bypasses RSI / Phase 2 feature warmup gates to speed local debugging
- `LLM_SHOW_DETAIL` is controlled by `custom/btc_agent/config.py`; when `true`, terminal output and completed-order / completed-period logs include the full LLM system/user prompt and raw response without enabling all `BTC_AGENT_DEBUG` output
- `BTC_AGENT_LOOP_INTERVAL` is controlled by `custom/btc_agent/config.py`; it currently controls the per-loop sleep interval in seconds
- `BTC_AGENT_MAX_TRADES_PER_PERIOD` default: `1`
- `MAX_LOSSES_PER_RUN` default: `4` in `custom/btc_agent/config.py`; the agent counts completed losing trades since launch and stops once that loss count reaches the configured threshold
- `BTC_AGENT_MIN_CONFIDENCE` optional override; otherwise the repo default comes from `DEFAULT_MIN_CONFIDENCE` in `custom/btc_agent/config.py`
- `BTC_AGENT_MAX_PRICE` is controlled by `custom/btc_agent/config.py`; it currently controls the maximum allowed entry price used by trade-quality gates
- `BTC_AGENT_MAX_SPREAD` optional override; otherwise the repo default comes from `DEFAULT_MAX_SPREAD` in `custom/btc_agent/config.py`, currently `0.10`
- `BTC_AGENT_MARKET_SLUG` for override/debugging/backtesting

Notes:

- `config.py` loads `.env` from the repo root, so local execution assumes a root-level `.env`.
- Non-secret threshold tuning is now beginning to move out of `.env` and into top-level defaults near the imports in [custom/btc_agent/config.py](/appl/agents/custom/btc_agent/config.py:1). Environment variables can still override those values, but the repo copy is now the primary reference for execution and prompt calibration defaults such as minimum confidence, ADX caution levels, market-win-chance veto levels, RSI veto thresholds, and required-velocity divisors.
- On the first loop and again at the start of each new 5-minute BTC period, the runtime now reloads `custom/btc_agent/config.py` in-process so edits to repo-defined non-secret tuning constants can take effect without restarting the bot.
- Order sizing now also follows that repo-defined pattern: `SHARES_PER_TRADE` in `custom/btc_agent/config.py` is the active fixed size for both paper and live BTC orders.
- `launch_btc_agent.sh` exports `.env` before starting Python so proxy variables such as `HTTP_PROXY` and `HTTPS_PROXY` are available to the full launch path.
- The active BTC-agent runtime normalizes `ALL_PROXY=socks5://...` to `socks5h://...` for `requests` traffic and to `socks5://...` for the OpenAI HTTPX client so Mullvad SOCKS routing works consistently.

## Current Behavior

What the BTC agent does today:

- Pulls the current BTC/USD spot price from a fallback chain of live providers, now preferring the Hermes/Pyth BTC reference endpoint the user specified, then falling back through Polymarket RTDS, Binance, Coinbase, and CoinGecko.
- The Hermes/Pyth BTC fetch path now mirrors the user-validated request shape more closely:
  - requests `parsed=true`
  - sends a browser-style `User-Agent`
  - reads the integer `price` plus `expo` fields from `parsed[0].price`
  - converts them to the human-readable BTC/USD price used by the agent
- Maintains an in-memory rolling price history during process lifetime only.
- Backfills enough recent BTC history on startup to support the Phase 2 indicator set, including the longer EMA(21) warmup.
- Even with that historical backfill, startup readiness now keys off fresh live samples for the visible RSI / Phase 2 warmup counters, so a restart still shows the familiar `RSI warmup incomplete (...)` and `phase 2 indicator warmup incomplete (...)` messages before trading is allowed.
- Approximates market-window open price using the earliest retained BTC sample inside the current 5-minute market window, not a true historical open fetched from a historical BTC data source.
- Computes 5-minute momentum and volatility from a trailing 5-minute BTC sample window, so analysis continues to reference recent cross-period history even immediately after a new market window begins.
- Computes a Phase 2 indicator set from the retained BTC history, including `RSI(9)`, `RSI(14)`, `EMA(9)`, `EMA(21)`, `ADX(14)`, `ATR(14)`, `rsi_speed_divergence`, `ema_alignment`, and `ema_cross_direction`.
- Applies a Phase 2.5 refinement layer that also computes `momentum_acceleration` and `oracle_gap_ratio`, and uses the faster `RSI(9)` reading as the primary trigger for aggressive `PARABOLIC_UP` / `PARABOLIC_DOWN` labeling.
- Falls back across multiple live BTC spot-price APIs first, then to the most recent in-memory BTC price sample when all configured live price requests fail during the current process lifetime, which prevents active paper-order reporting from aborting immediately on a single-provider rate-limit response after recent successful samples.
- Uses the current 5-minute BTC Up/Down slug by timestamp alignment, unless overridden.
- For BTC 5-minute up/down markets, token-to-outcome discovery now treats `clobTokenIds[0]` as the YES/`UP` token and `clobTokenIds[1]` as the NO/`DOWN` token, matching the user-validated `getlimitfull.py` mapping rather than relying on weaker token metadata ordering.
- Performs a startup IP geolocation check and refuses to run unless the current public IP resolves to an allowed country, currently Indonesia or Mexico.
- Bypasses the startup geolocation check when `LLM_CONNECTION_DEBUG=true` so LLM/provider connectivity can be tested from non-allowed locations.
- Respects standard proxy environment variables such as `HTTP_PROXY` and `HTTPS_PROXY` for outbound requests when they are exported in the shell or defined in the repo `.env`.
- Routes outbound BTC-agent requests through `ALL_PROXY` when configured, including geolocation, BTC spot pricing, Polymarket API lookups, and LLM calls.
- For Polymarket HTTP endpoints, the shared network layer now retries once without the proxy on connect/read timeout when proxy routing is enabled, which helps recover from intermittent SOCKS timeouts against `*.polymarket.com`.
- Allows proxy routing to be disabled globally with `USE_PROXY=false`, which causes the agent to use direct connections for shared HTTP/LLM requests even if proxy environment variables are present.
- Supports a dedicated `LLM_CONNECTION_DEBUG=true` mode that runs only LLM diagnostics, prints the active connection settings, prints the full system/user prompt for each probe, sends a small LLM probe first, runs a direct Google connectivity probe after LLM failures, then sends a predefined full-size trading-style prompt before exiting without touching geolocation, balances, market lookup, warmup, or trading execution.
- The old `priceToBeatDebug*.txt` runtime file generation path has been removed; debug and live runs no longer create or clear those files.
- Uses the configured AI engine with JSON output to decide `UP`, `DOWN`, or `NO_TRADE`.
- Prints the current market `price_to_beat` in the BTC-agent output and includes that same period baseline in the LLM decision prompt, now preferring Vatic's BTC 5-minute timestamp target API and only falling back to Polymarket page / `_next/data` parsing when that external target lookup is unavailable.
- Pulls Gamma `outcomePrices` from the active market payload only as an initial discovery fallback, then refreshes `up_market_probability` / `down_market_probability` from the Polymarket CLOB market websocket using the same token-mapped `price_change.best_ask` fields as the local `getlimitfull.py` script, so terminal output, completed-period logs, completed-order logs, and prompt inputs track the same live probabilities shown on Polymarket more closely.
- Prints `up_spread` and `down_spread` in the per-tick `Market:` block so each loop shows the current book width for both sides even when verbose quote snapshots are not being emphasized.
- Derives `best_bid`, `best_ask`, `spread`, and `spread_bps` from the executable quote pair first (`SELL` quote as bid, `BUY` quote as ask), only falling back to raw `/book` extremes when quote-side prices are unavailable. This avoids the old ghost-spread problem where thin 5-minute books surfaced fake `0.01 / 0.99` bounds and triggered toxic-liquidity rejections.
- Uses a one-shot synchronous websocket scrape for that per-tick probability refresh and closes it immediately after both side probabilities arrive, avoiding the older `asyncio.run()` / websocket close-handshake delay that could stretch a nominal 5-second sleep into 10-15 seconds.
- Refreshes those live market probabilities only once per loop tick; the main loop now reuses the market object’s already-refreshed `up_market_probability` / `down_market_probability` values instead of opening a second websocket connection.
- The live market-probability path is now authoritative: if the CLOB market websocket does not provide `best_ask` values for both outcome tokens, the refresh path raises instead of silently inventing or substituting probabilities.
- Retries LLM API calls across configurable attempts using a single per-attempt timeout, logs each attempt result to stdout, and converts repeated failures into a `NO_TRADE` so the loop can move on to the next tick.
- Computes a reference price from quote, midpoint, last trade, and order book data.
- Reuses a single decision-time token quote snapshot for both the printed `UP/DOWN (with decision)` block and the paper execution gate so those logs cannot diverge within one loop tick.
- Uses `USE_RECOMMENDED_LIMIT` to choose whether submit checks and execution use the recommended snapped limit or the raw target limit; when disabled, recommended limit is still computed for visibility but is not used as a gating/execution factor.
- When `USE_RECOMMENDED_LIMIT=false`, the agent still collects internal CLOB quote/book context for LLM reasoning and completed-order logging, but it no longer prints the pre-trade `UP` / `DOWN` quote snapshots or uses recommended-limit gating.
- Skips LLM decision calls entirely when both the current `UP` and `DOWN` quote snapshots are already not safe to submit, preserving AI API calls when neither side is actionable.
- Provides the LLM with time remaining, window delta, and current `UP` / `DOWN` ask prices so the model can apply EV- and timing-based rules for late-window decisions.
- Provides the LLM with the Phase 2 trend-strength and normalization fields as well, including `RSI(9)`, `RSI speed divergence`, `EMA` alignment/cross direction, `ADX(14)`, `ATR(14)`, and the volatility-normalized / required-velocity context derived from the current target gap.
- To reduce transport timeouts, the live decision path now uses a trimmed minimal LLM payload for both OpenAI Realtime and Gemini instead of the older verbose prompt, while preserving the active Phase 2.x guardrails.
- The LLM prompt now also includes `momentum_acceleration`, `last_10_ticks_direction`, and follows stricter ADX guidance:
  - if `ADX(14) > 35`, do not trade against the trend
  - if `ADX(14) > 45`, treat the move as potentially exhausted and avoid late trend-chasing
- The LLM prompt now treats `time_remaining_seconds` as the authoritative clock:
  - final 10 seconds means `time_remaining_seconds < 15`
  - `time_remaining_seconds > 240` is treated as an early-window Discovery Phase where high-confidence trades should be rare unless trend intensity is extreme
  - `Window Delta` is explicitly defined as the change from the market window open price and must not be confused with `DISTANCE_FROM_STRIKE_*`, `MARKET_WIN_CHANCE_*`, or `oracle_gap_ratio`
  - when upstream `start_ts` / `end_ts` are stale, the prompt now prefers the canonical slug timestamp plus 300 seconds so the model sees the same effective clock as execution
- The LLM prompt now treats `DISTANCE_FROM_STRIKE_PCT` as the source of truth for whether `UP` or `DOWN` is currently winning relative to the price to beat, while `window_delta_pct` is treated only as recent drift from the market-window open.
- `DISTANCE_FROM_STRIKE_USD` and `DISTANCE_FROM_STRIKE_PCT` are now computed directly from the raw BTC feed price minus the period `price_to_beat`; synthetic Polymarket-implied oracle price is included only as separate market-context diagnostics and must not replace the physical strike gap.
- The live decision prompts now omit `window_delta_pct` entirely from the active minimal decision payload and instead emphasize:
  - `DISTANCE_FROM_STRIKE_USD`
  - `DISTANCE_FROM_STRIKE_PCT`
  - `MARKET_WIN_CHANCE_UP`
  - `MARKET_WIN_CHANCE_DOWN`
- The prompt explicitly tells the model not to confuse strike-distance fields with market-win-chance fields, which closes the Phase 2.82 “logic loop hallucination” where the model treated probability and price-distance as the same signal.
- The LLM prompt now also passes a drift-adjusted effective BTC price, derived from the implied Polymarket oracle price when available, and tells the model to use that effective price instead of the raw feed price when reasoning about strike distance.
- The LLM prompt uses `velocity_30s` only as a micro-momentum entry-timing signal, which is intended to prevent baseline-confusion reversals where the model mistakes a short dip for being below the strike.
- The LLM prompt now uses a regime-aware market-win-chance sanity rule instead of the older flat quote veto:
  - if the chosen side `MARKET_WIN_CHANCE` is below `0.15` and `15 <= time_remaining_seconds < 120`, prefer `NO_TRADE`
  - under the final 15 seconds, only fade that consensus on a clear reversal
- The LLM prompt now also treats early-window caution as ADX-aware rather than flat:
  - if `time_remaining_seconds > 240` and `ADX < 15`, stay cautious
  - if `ADX > 30`, prioritize the trend over the elapsed time
- The LLM prompt now includes a momentum-trap exhaustion rule:
  - if `ADX > 50` and `RSI(9) > 75`, treat that as exhaustion rather than fresh breakout strength and avoid opening a new position
- The LLM prompt now includes an early-window near-strike caution rule:
  - if `time_remaining_seconds > 180` and `DISTANCE_FROM_STRIKE_USD < 0.5 * volatility_5m`, prefer `NO_TRADE`
  - if `time_remaining_seconds > 240` and `ADX > 30`, prioritize the trend over elapsed time
- The LLM prompt now includes `momentum_alignment`, which is `True` when `velocity_15s`, `velocity_30s`, and `momentum_1m` all point the same direction; when it is `True`, the model is encouraged to treat clarity as high and trade with the trend.
- The LLM prompt now includes an in-the-money confidence boost rule: if `DISTANCE_FROM_STRIKE_USD` is beyond `0.5 * ATR` in the chosen direction, the model may raise confidence by `0.15`.
- The LLM prompt now explicitly disambiguates market probability from physical strike distance: a `MARKET_WIN_CHANCE` of `65%` is an aggressive consensus signal, while even a `$1.00` physical `DISTANCE_FROM_STRIKE_USD` can justify `~95%` confidence when fewer than 15 seconds remain.
- The LLM prompt includes a sign-consistency rule: if `DISTANCE_FROM_STRIKE_PCT` is positive and the model chooses `DOWN`, or negative and it chooses `UP`, confidence should stay below `0.50` unless trend exhaustion is genuinely clear.
- The LLM prompt now also includes an RSI directional sanity rule: if `RSI(9) < 30`, it should not choose `DOWN`; if `RSI(9) > 70`, it should not choose `UP` unless `ADX > 30`, and values above `RSI(9) > 85` are treated as explicit exhaustion risk even in strong trends.
- The LLM prompt and execution layer now both suspend the normal `UP` RSI veto when `rsi_speed_divergence > 5` and `ADX > 35`, so accelerating RSI in a strong trend is treated as a parabolic continuation signal instead of automatic exhaustion.
- The LLM prompt now also tells the model to lower confidence when `rsi_speed_divergence` is negative while price is still moving up, which is intended to catch weakening continuation setups before they become late-window fades or weak breakout chases.
- The LLM prompt now also carries a late-window ITM maintenance rule: if a side is already in the money and fewer than 60 seconds remain, generic RSI exhaustion fears should be deprioritized unless a genuinely large counter-trend spike is occurring.
- The Phase 2.7 prompt layer now also instructs the model to prefer `NO_TRADE` when:
  - the market is too close to call, meaning `abs(gap_to_target_usd) < 0.2 * volatility_5m` with more than 60 seconds remaining
  - it wants `UP` while the `UP` quote is below `0.45`
  - `RSI(9) > 85` and BTC is already above the strike, unless it is inside the final 15 seconds and continuation is exceptionally clear
- Skips LLM decisioning and execution during the last 60 seconds of the current 5-minute market window and only continues collecting trend data for the upcoming period.
- Prints the exact execution snapshot used by the paper-trade path, including the calculated `reference_price`, `target_limit_price`, and `recommended_limit_price`.
- Executes paper trades by default and can submit live Polymarket buy orders through `agents/polymarket/polymarket.py` when `USE_PAPER_TRADES=false`.
- Submits live orders with a configurable maker fee rate from `BTC_AGENT_LIVE_FEE_RATE_BPS`, defaulting to `1000` bps to match the current BTC Up/Down market requirement observed during live submission attempts.
- Sizes paper and live orders from `SHARES_PER_TRADE` in `custom/btc_agent/config.py`, using a fixed share count for both paper and live orders.
- The current repo-visible sizing default is `SHARES_PER_TRADE = 3`, so both paper and live orders now use that fixed reduced exposure unless the code default is changed again.
- Before live BUY submission, the agent quantizes share size to a Polymarket-compatible precision so the quote-side amount stays within the exchange’s 2-decimal maker-amount constraint while still respecting the 4-decimal taker-size limit.
- That live BUY quantization now guarantees a minimum positive valid quantum when the configured size is non-zero, preventing edge-case rounding from collapsing a tiny but valid order to zero shares.
- Rejects live submissions cleanly when the configured `SHARES_PER_TRADE` cannot satisfy the venue minimum order size instead of silently scaling or overriding size.
- Applies a regime-aware execution-side market-win-chance veto before submission, using Gamma probability first and falling back to CLOB quote only if Gamma is unavailable:
  - if the chosen side `MARKET_WIN_CHANCE < 0.15` and `15 <= time_remaining_seconds < 120`, the trade is rejected as a low-probability reversal attempt
- Applies a velocity/volatility sanity veto before submission only when the chosen side is currently out of the money versus the strike: if that OTM trade requires `required_velocity_to_win > volatility_5m / 7.5`, it is rejected as a mathematically implausible move. If the chosen side is already in the money, the velocity veto is skipped.
- Applies a Gamma/CLOB consensus-gap veto before submission: if `abs(effective_confidence - market_implied_probability) > 0.50`, the trade is rejected as a hallucinated edge against market consensus.
- Applies a regime-aware minimum-confidence veto before submission:
  - base floor now defaults to `DEFAULT_MIN_CONFIDENCE` through the top-level tuning defaults in `custom/btc_agent/config.py`, while still allowing a `BTC_AGENT_MIN_CONFIDENCE` env override
  - in the Discovery Phase with more than 180 seconds remaining, the operational floor now uses `DISCOVERY_MIN_CONFIDENCE`, currently `0.85`
  - in the mid-window band, the operational floor now falls back to the base configured minimum-confidence value instead of using the older ADX-based relaxed floor
  - in the final 60 seconds, the operational floor is tightened to `0.70`
- The execution layer now computes an `effective_confidence` before gating:
  - if the chosen side is already winning against the strike by more than `0.5 * ATR(14)` in the chosen direction
  - the confidence is boosted by `+0.15` before floor / edge / consensus-gap checks
- Execution timing now also prefers the canonical slug timestamp plus 300 seconds over stale upstream `start_ts` / `end_ts`, so late-window vetoes and FOK logic use the same boundary the logs print in `mm:ss`.
- Applies a configured spread veto before thin-liquidity checks: if `snapshot.spread > max_spread`, the trade is rejected. The repo default is now `0.10`, which is intentionally looser than the earlier `0.06` setting.
- Applies a dynamic “too close to call” strike buffer before submission instead of a flat `0.2 * volatility_5m` rule:
  - during the first 180 seconds the buffer uses `DYNAMIC_STRIKE_BUFFER_MULTIPLIER`, currently `0.40 * volatility_5m`
  - after that it decays using the existing early/late window buffer multipliers
  - this is intended to reduce early near-strike chop losses without over-blocking stronger late-window entries
- Applies a Phase 2.7 quote-price divergence veto for `UP`: if the bot wants `UP` but the `UP` quote is below `0.45`, the trade is rejected because the market is not confirming the breakout.
- Applies a Phase 2.7 RSI ceiling veto for `UP`: if `RSI(9) > 85` while BTC is already above the strike, the trade is rejected as an exhaustion-risk breakout chase.
- Applies RSI directional vetoes before submission:
  - if `RSI(9) < 30`, `DOWN` is rejected as bottom-chasing
  - if `RSI(9) > 70`, `UP` is rejected unless `ADX > 30`
  - if `RSI(9) > 85` while BTC is already above the strike, `UP` is rejected as an exhaustion-risk breakout chase
- Applies a hard exhaustion cap before the softer RSI vetoes:
  - if `ADX(14) > 50` and `RSI(9) > 80`, new `UP` entries are rejected as overbought trend exhaustion
  - if `ADX(14) > 50` and `RSI(9) < 20`, new `DOWN` entries are rejected as oversold trend exhaustion
- Applies a late ITM quote cap: if the chosen side is already in the money, fewer than 60 seconds remain, and the buy quote is above `0.85`, the trade is rejected as poor risk/reward.
- Applies momentum-trap RSI vetoes before submission:
  - if the bot wants `UP` while BTC is still below the strike and `RSI(9) > 60`, the trade is rejected as a late breakout chase below the settlement line
  - if the bot wants `UP` with more than 200 seconds remaining and `RSI(9) > MAX_EARLY_WINDOW_RSI`, currently `65`, the trade is rejected as an early-window momentum trap
- Suspends the normal `UP` RSI veto during parabolic continuation only when both of these are true:
  - `rsi_speed_divergence > 5`
  - `ADX(14) > 35`
- Decision-time recommended-limit gating now allows up to a 4-tick adverse move instead of 2 ticks when `volatility_5m` is in the `extreme` regime, which helps preserve valid entries during fast quote drift.
- Applies a Discovery strike-buffer veto during the first 60 seconds of a new window: if `time_remaining_seconds > 240` and the BTC price is within `0.2 * ATR` of the strike, the trade is rejected unless momentum alignment is fully positive and ADX already shows strong trend strength.
- Tracks in-memory active orders for the current 5-minute market window and prints each order’s target BTC level plus whether the position is currently winning, losing, or tied.
- If a live order has been submitted but no fill is confirmed yet, the runtime now shows `order_<n>_position_state = UNFILLED` instead of `PENDING_FILL`, so the terminal output clearly distinguishes an accepted order from a confirmed fill.
- Writes a per-slug order-tracking file under `completed_orders/` for each executed order, appending one status snapshot per tick plus the pre-order tick history that led into the trade.
- Writes `completed_order_attempt_<slug_timestamp>.txt` when a directional trade is rejected before execution or live submission fails:
  - in paper mode, these files represent pre-execution validation rejections and explicitly mark `order_submission_attempted=false`
  - in live mode, they represent actual submission failures or final live-attempt rejections and explicitly mark `order_submission_attempted=true`
  - in both modes, the file preserves the exact `attempt_reason` so trade review can see why the bot tried and why it did not execute
- Finalized executed-order filenames now encode both trade outcome and actual period result direction:
  - `completed_order_win_up_<slug_timestamp>.txt`
  - `completed_order_loss_down_<slug_timestamp>.txt`
  - and `...-<trade_num>.txt` suffixes still apply when multi-trade-per-period mode is enabled
- Live fill confirmation now prefers explicit Polymarket order-status data (`get_order(order_id)`) plus trade-resolution lookups instead of inferring fills from generic submission-response fields.
- If a live order submission is accepted but no actual fill price is confirmed yet, the agent now tracks it as an unfilled live order and rechecks Polymarket status on later ticks and at period rollover.
- On later ticks, if that accepted live order is still unfilled, the bot now attempts one live maintenance retry:
  - cancel the outstanding order through the Polymarket client
  - fetch a fresh current quote snapshot for the same token
  - re-submit the same side and size at the latest submission-limit probability price
  - if the retry still does not fill, the order remains tracked as `UNFILLED`
- If that tracked live order reaches period close without a confirmed fill price, the finalized filename is neutral and does not carry a win/loss or direction label:
  - `completed_order_unfilled_<slug_timestamp>.txt`
- If a confirmed filled order shows realized slippage above `SLIPPAGE_COOLDOWN_THRESHOLD_BPS`, currently `500` bps, the bot now triggers a trade cooldown for `SLIPPAGE_COOLDOWN_SECONDS`, currently `300` seconds, before taking another order.
- While an executed order is still unresolved, its working log file remains `completed_order_<slug_timestamp>.txt` and is only renamed to the finalized `completed_order_<win/loss>_<up/down>_<slug_timestamp>.txt` form once the period result is known.
- Preserves every analyzed 5-minute window under `completed_orders/completed_period_<slug_timestamp>.txt` while the period is still unresolved, and finalizes resolved no-trade period files as:
  - `completed_period_up_<slug_timestamp>.txt`
  - `completed_period_down_<slug_timestamp>.txt`
  when a final BTC resolution price is available on rollover.
- While a period is still live, the working analysis file is `pending_period_<slug_timestamp>.txt`; that file is finalized on rollover or graceful exit so period analysis is not stranded.
- Completed-order and completed-period loop entries now include both raw remaining seconds and a `mm:ss` market-time-remaining field for easier late-window analysis.
- Completed-order and completed-period logs now also carry Phase 2.7 price-lag diagnostics in the regime fingerprint:
  - `implied_oracle_price`
  - `feed_drift_usd`
  - `last_10_ticks_direction`
- Live submission-time exceptions no longer abort the full bot process. Instead, the agent writes `completed_orders/failed_order_<slug_timestamp>.txt` containing the copied per-period context collected so far plus the live submission failure reason, then continues to the next loop tick.
- Completed-period logs now also preserve `up_market_probability` and `down_market_probability` from the live per-loop market view, which now prefers the token `best_ask` values that match Polymarket’s displayed probabilities.
- Pre-order decision logging now also records `effective_confidence`, which is the regime-adjusted operational confidence after the winning-advantage boost and the dynamic minimum-confidence floor are applied.
- Pre-order and attempt logs now also record:
  - `min_confidence`
  - `veto_threshold_delta`
  - `llm_raw_reasoning`
  - `market_impact_ratio` when size and top-of-book depth are both known
- `last_10_ticks_direction` is now a filtered string of meaningful recent `U` / `D` moves only; flat ticks and sub-threshold micro-jitter are omitted so it better reflects genuine short-term direction instead of noisy tape artifacts.
- On period rollover, the pending period log now uses the next market’s resolved `price_to_beat` even when no trade was executed in the previous period, so no-trade period files also finalize to an `_up_` or `_down_` filename much more consistently.
- On graceful exit (`q` or `KeyboardInterrupt`), the current slug’s `pending_period_<timestamp>.txt` is finalized into `completed_period_<timestamp>.txt` so pending analysis files are not stranded.
- Evaluates paper-order win/loss status against the market-period settlement reference, preferring Polymarket’s parsed threshold and otherwise falling back to the closest retained BTC sample at the start of the 5-minute period rather than the trade-entry BTC price.
- Enforces `BTC_AGENT_MAX_TRADES_PER_PERIOD` per 5-minute market slug; current production usage assumes `1`, so once that limit is reached the remaining loop ticks in the slug skip new trade evaluation and only continue status/data tracking until the next market window begins.
- Enforces `MAX_LOSSES_PER_RUN` across the full process session as a completed-loss stop; once that many trades have actually settled as losses, the agent exits.
- When `BTC_AGENT_DEBUG=false`, suppresses most verbose diagnostics and only prints a compact subset of geolocation, balances, quote snapshots, BTC features, LLM decision fields, and final paper execution fields.
- When `BTC_AGENT_DEBUG=true`, the agent also prints the exact LLM prompt text to terminal output and writes that same prompt into the pending-period / completed-order logs for post-trade review.
- When `LLM_SHOW_DETAIL=true`, the agent also prints and stores the exact LLM prompt/response details without requiring full debug mode.
- The raw LLM response text is preserved for successful decisions regardless of debug mode:
  - it is printed to terminal output as `LLM raw response:`
  - it is written into pending-period / completed-order logs between `llm_raw_response_start` / `llm_raw_response_end`
  so prompt/parse discrepancies can be audited directly.
- Phase 3 execution/microstructure metrics are now captured on executed orders and written into completed-order logs:
  - `quoted_price_at_entry`
  - `actual_fill_price`
  - `realized_slippage_bps`
  - `order_latency_ms`
  - `book_depth_at_fill`
  - `shares_requested`
- In paper mode, those same Phase 3 metrics are also carried into `completed_order_attempt_*` files for pre-execution rejections when they can be derived from the simulated quote snapshot, so paper-mode analysis can still distinguish signal rejection from thin-book conditions.
- Phase 3.1 refinement: vetoed directional attempts now preserve the bot’s intent-time quote context too, so `completed_order_attempt_*` files capture the quote and top-of-book depth the bot wanted to trade against even when a pre-execution guardrail blocks the trade.
- In non-debug mode, account balances print only on the first loop iteration and again at the start of each new 5-minute market period.
- Stops the process before live order submission when the account does not have enough `cash_balance_pusd` to cover the configured live trade size at the recommended limit price plus the estimated fee buffer.
- Stops the process when `cash_balance_pusd` falls below `MINIMUM_WALLET_BALANCE`, so no further execution occurs once the configured wallet floor is breached.
- Retrieves Polygon pUSD trading cash balances through a configurable ordered RPC list with public fallback endpoints, and also reports legacy USDC.e separately for migration visibility.
- Retrieves Polymarket portfolio value separately from the on-chain cash balance lookup so one failure does not suppress the other.
- Approves or rejects a trade based on confidence, entry caps, quote drift, and in live mode also account cash availability.
- Prints diagnostics for balances, quotes, features, decision, and simulated execution.
- The `Features:` block prints a single `btc_price`, which now comes from the Hermes/Pyth reference feed first and only falls back to the older providers if that feed is unavailable.

What it does not do yet:

- Persist price history or trading state across restarts.
- Use a true market-window open from historical BTC data.
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
- Active paper orders and per-period trade counts are also process-local only, so a restart clears them immediately.
- The live-trading path supports proxy-wallet execution when `POLYMKT_PROXY_ADDRESS` is set by initializing the upstream Polymarket CLOB client with `signature_type=2` and the proxy address as the funder.
- The market lookup assumes the current slug format and first market in the event response remain stable.
- The decision path depends on a single LLM response and does not yet validate response quality beyond JSON parsing and basic coercion.
- Network/API failures are still only partially hardened; balance lookups now degrade more cleanly, but other external calls remain single-point dependent.
- Multi-provider spot-price fallback still only has cached-price protection when the current process already has a recent BTC price in memory; a cold start where all configured BTC price providers fail immediately can still abort the loop.
- Live order submission depends on the wallet already having the required Polymarket / CTF approvals; the custom BTC flow does not auto-run approvals before submitting a live order.
- Live fee requirements can vary by market; `BTC_AGENT_LIVE_FEE_RATE_BPS` is currently only an estimated cash-buffer input for the BTC agent, while live order creation itself now relies on the Polymarket CLOB V2 SDK fee model.
- Live minimum order requirements can also vary by venue or product; `BTC_AGENT_LIVE_MIN_ORDER_USD` is currently a configurable static target used to scale live order size upward based on the observed Polymarket rejection threshold.
- There are no BTC-agent tests for market parsing, pricing, decision normalization, or paper-trade gating.
- The inherited repo still contains placeholder tests and unused upstream surfaces, which can mislead future work if not distinguished from the active BTC path.

## Ongoing Development Guidance

When modifying this repo, future Codex runs should:

- Treat `custom/btc_agent/` as the primary application area unless the task is explicitly about upstream framework code.
- Preserve paper-trading-only behavior unless the user explicitly requests live trading changes.
- Avoid changing unrelated upstream modules just because they exist; keep BTC-specific logic isolated where practical.
- Add or update tests when changing pricing, market selection, decision parsing, or execution gating.
- Update this file whenever architecture, runtime flow, env vars, or user-facing behavior changes.

## Improvement Roadmap

The BTC agent is now in a measurement-and-refinement phase. Before making larger strategy changes, each phase should be completed in order so the team has the evidence needed to justify the next step.

### Phase 1: Logging And Regime Capture

Goal:

- Capture enough per-tick context to explain why a trade won or lost.

Current status:

- Complete, with follow-up metric and calibration corrections applied after review.
- Completed-order logs now include richer per-tick feature data, active-order snapshots, internal CLOB book context, and a deterministic `regime_fingerprint`.
- Phase 1 now also captures micro-momentum and reachability context:
  - `velocity_15s`
  - `velocity_30s`
  - `consecutive_flat_ticks`
  - `consecutive_directional_ticks`
  - `required_velocity_to_win`
  - `volatility_to_gap_ratio`
  - `orderbook_imbalance_snapshot`
  - `llm_veto_flags`
  - `time_decay_confidence`
- Deterministic regime labeling now includes `PARABOLIC_UP` / `PARABOLIC_DOWN` so extreme RSI in strong momentum is treated as trend continuation context rather than automatic mean-reversion.
- Even when `USE_RECOMMENDED_LIMIT=false`, the agent still fetches internal `UP` / `DOWN` book context for logging and LLM reasoning, while continuing to suppress the old pre-trade quote snapshot console output.
- The `consecutive_directional_ticks` counter now ignores short flat-tick interruptions, near-duplicate same-price samples, and small counter-moves below a 0.01% BTC-price reversal threshold, so it better reflects real one-way streaks for falling-knife / exhaustion detection.
- `liquidity_regime` now explicitly marks very wide books as `THIN_LIQUIDITY` when `spread_bps > 150`, which keeps obviously toxic 5-minute books from being treated as merely "low" liquidity.

Data required:

- `btc_price`
- `delta_prev_tick`
- `momentum_1m`
- `momentum_5m`
- `volatility_5m`
- `window_open_price`
- `delta_from_window_pct`
- `trailing_5m_open_price`
- `delta_from_5m_pct`
- `rsi_14`
- `time_remaining_seconds`
- `threshold_gap_usd`
- `threshold_gap_pct`
- `strike_delta_pct`
- `up_market_probability`
- `down_market_probability`
- `best_bid`
- `best_ask`
- `best_bid_size`
- `best_ask_size`
- `spread`
- `spread_bps`
- `top_level_book_imbalance`
- `imbalance_pressure`
- `velocity_15s`
- `velocity_30s`
- `consecutive_flat_ticks`
- `consecutive_directional_ticks`
- `required_velocity_to_win`
- `volatility_to_gap_ratio`
- `orderbook_imbalance_snapshot`
- `llm_veto_flags`
- `time_decay_confidence`
- final order outcome classification using the resolved closing price

Completion criteria:

- Completed-order files consistently explain wins and losses without requiring manual reconstruction from console logs.
- Rollover finalization uses the next slug’s `price_to_beat` or an equivalent resolved closing price rather than a stale live spot sample.
- `top_level_book_imbalance`, `imbalance_pressure`, and `liquidity_regime` should no longer be routinely null/unknown in Phase 1 win/loss files.
- `consecutive_directional_ticks` should move with genuine one-way price streaks instead of resetting on flat ticks or tiny counter-moves.

### Phase 2: Indicator Expansion

Goal:

- Add faster and more context-aware indicators so the agent can distinguish parabolic continuation, mean reversion, and ranging behavior.

Current status:

- In progress, with the first Phase 2 indicator pass implemented, a Phase 2.5 refinement layer added after reviewing the first Phase 2 losses, a Phase 2.6 cleanup pass applied after validating the first enriched win/loss files, and a Phase 2.82 prompt/gating calibration pass added after identifying a low order-rate “logic loop hallucination.”
- The BTC feature set now includes:
  - `RSI(9)` alongside `RSI(14)`
  - `EMA(9)` and `EMA(21)`
  - `ema_alignment`
  - `ema_cross_direction`
  - `ADX(14)`
  - `ATR(14)`
  - `rsi_speed_divergence`
  - `volatility_normalized_gap` in the regime fingerprint
  - `momentum_acceleration`
  - `oracle_gap_ratio`
- The regime builder now prioritizes the fast RSI signal when deciding whether the environment is `PARABOLIC_UP` or `PARABOLIC_DOWN`.
- The regime builder is also now strike-aware, so if BTC is materially above or below the current `price_to_beat`, the trend label will not flip to the opposite side just because the very recent window delta is slightly negative or positive.
- `RSI(9)` and `RSI(14)` are now computed independently from the trailing price window, so the fast RSI can diverge from the slow RSI instead of silently collapsing onto the same value.
- Feature readiness now treats the RSI / Phase 2 warmup counters as live-run counters rather than trusting startup backfill alone, so the bot must accumulate fresh post-launch samples before those indicator gates become ready.
- The LLM prompt paths now explicitly carry the advanced Phase 2.5 / 2.6 fields, including:
  - `momentum_acceleration`
  - `trend_intensity` / `ADX(14)`
  - `oracle_gap_ratio`
  - `DISTANCE_FROM_STRIKE_PCT`
  - authoritative `time_remaining_seconds` / Discovery Phase guidance
  - explicit `MARKET_WIN_CHANCE_*` vs `DISTANCE_FROM_STRIKE_*` field separation
  - regime-aware market-win-chance gating for discovery / mid-window periods
- Phase 2.82 also changed the operational gating logic:
  - `effective_confidence` now includes a winning-advantage boost when the chosen side is already in the money and the Gamma market agrees
- the confidence floor is now regime-aware instead of always flat
- the old static quote veto was replaced with a Gamma-first market-win-chance veto tuned for higher order rate earlier in the window
- Phase 3.2 refined that further by removing the old discovery-phase `<0.10` veto, relaxing the trend floor to `ADX > 30` during the first 180 seconds, and adding `momentum_alignment` plus `veto_threshold_delta` / `market_impact_ratio` logging for order-rate analysis

Implemented additions:

- `RSI(9)` alongside the current `RSI(14)`
- `EMA(9)` and `EMA(21)` plus cross direction
- `ATR(14)` or similar volatility-normalized move measure
- `ADX(14)` for trend-strength detection
- `rsi_speed_divergence`
- `volatility_normalized_gap`
- `momentum_acceleration`
- `oracle_gap_ratio`

Still optional / not implemented yet:

- Bollinger Band Width or equivalent squeeze detector

Data required:

- rolling BTC history with enough depth for the new lookbacks
- per-tick logging of all new indicator values
- regime comparison between old and new indicator interpretations

Completion criteria:

- New indicators are computed reliably on every eligible tick.
- Completed-order files show the new indicator values for later post-trade review.
- The LLM prompt and completed-order logs both contain the new Phase 2 values so the expanded indicator set can be evaluated against real wins and losses.
- Phase 2.5 loss review fields should be present as well, especially `momentum_acceleration`, `oracle_gap_ratio`, and the fast-RSI-driven parabolic regime labels.
- Thin-liquidity books with `spread_bps > 150` should be classified as `THIN_LIQUIDITY` and rejected before they become noisy Phase 3 execution data.

### Phase 3: Execution And Microstructure Features

Goal:

- Improve trade-quality filtering with execution-aware data rather than price-only context.

Current status:

- Started, with the first execution-metrics pass implemented and a Phase 3.3 calibration pass now moving the main order-rate and veto thresholds into repo-visible config defaults instead of burying all of them in `.env`.
- Executed orders now capture and log:
  - `quoted_price_at_entry`
  - `actual_fill_price`
  - `realized_slippage_bps`
  - `order_latency_ms`
  - `book_depth_at_fill`
  - `shares_requested`
- Paper trades use the chosen submission limit as the simulated fill price and record zero submission latency.
- Live trades now measure round-trip submit latency around the actual Polymarket order call and attempt to resolve average fill price from the live response first, then from the Polymarket trades endpoint when an order id is available.
- Completed-order logs now preserve those fields across `PLACED`, `ACTIVE`, and `COMPLETED` lifecycle entries so later analysis can compare prediction quality against execution quality.
- Rejected directional attempts now also preserve `quoted_price_at_entry`, `book_depth_at_fill`, and derived `shares_requested` before veto logic returns, so Phase 3 analysis can inspect trades the bot wanted to place but did not execute.
- Phase 3.3 also relaxed and centralized several order-rate controls:
  - default minimum confidence is now `0.60`
  - Discovery Phase caution now keys off `ADX < 15` instead of `ADX < 20`
  - the velocity/volatility veto now uses `volatility_5m / 15`
  - the execution edge floor now defaults to `0.02`
  - the stronger-trend `UP` RSI veto now becomes dynamic, using `RSI > 70` normally and only hard-blocking continuation at `RSI > 85` when `ADX > 30`
  - these values now live as top-level non-secret defaults in `custom/btc_agent/config.py` and can still be overridden from env when needed

Data required:

- pre-trade quote snapshot
- execution snapshot
- final fill price
- tick-by-tick spread and imbalance history while an order is active
- quoted vs actual fill delta
- order-submit round-trip latency
- top-of-book depth available at the time of the order

Completion criteria:

- Loss analysis can separate bad prediction from bad execution.
- The logs clearly show whether a loss came from signal quality, spread expansion, or thin liquidity.
- After enough real trades accumulate, wins/losses can be compared directly against `realized_slippage_bps`, `order_latency_ms`, and `book_depth_at_fill` to decide whether a future depth filter or network-speed mitigation is warranted.

### Phase 4: Prompt And Decision Structure

Goal:

- Improve the LLM’s reasoning quality before any fine-tuning by making the prompt explicitly regime-aware and more structured.

Planned changes:

- enforce reasoning structure for:
  - regime detection
  - indicator reconcilement
  - threshold proximity
  - final decision/confidence
- prefer concise structured reasoning over generic TA narration
- add explicit handling for momentum traps, oversold bounces, and parabolic continuations

Data required:

- Phase 1 and Phase 2 logs
- a representative sample of wins and losses with completed-order files

Completion criteria:

- LLM reasons are more specific and consistent across similar market regimes.
- Fewer losses are attributable to obvious indicator misreads like fading a strong parabolic move.

### Phase 5: Multi-Timeframe Context

Goal:

- Add higher-timeframe directional context so the 5-minute trigger is not acting in isolation.

Planned additions:

- 15-minute trend context
- 1-hour trend context
- alignment fields such as:
  - higher-timeframe trend direction
  - price vs higher-timeframe averages
  - higher-timeframe volatility state

Data required:

- higher-timeframe BTC samples or derived aggregates
- logging of multi-timeframe values into completed-order files

Completion criteria:

- The LLM can explicitly reference short-term vs higher-timeframe alignment in its decision logic.

### Phase 6: Dataset Labeling And Model Evaluation

Goal:

- Convert completed-order logs into a usable training and evaluation dataset.

Planned workflow:

- separate wins and losses into labeled trajectories
- attach final close outcome and relevant forward labels
- produce structured examples for later prompt testing or fine-tuning

Data required:

- completed-order files with full pre-order and active-order history
- deterministic final outcome price
- original decision reasoning and execution context

Completion criteria:

- A consistent labeled dataset exists for prompt iteration and later fine-tuning experiments.

### Phase 7: Fine-Tuning / Shadow Testing

Goal:

- Only after the data pipeline and prompt structure are stable, test a specialized model or shadow-decision path.

Planned workflow:

- create a shadow decision engine using the labeled dataset
- compare production vs shadow decisions without executing shadow trades
- measure improvements before promotion

Data required:

- Phase 6 labeled dataset
- clear evaluation metrics such as win rate, drawdown behavior, and regime-specific accuracy

Completion criteria:

- Shadow results consistently outperform the existing decision path before any promotion to live usage.

## Phase Tracking Rules

Future Codex runs should:

- treat the phases above as the default improvement sequence unless the user explicitly reprioritizes
- avoid jumping to fine-tuning before the logging and indicator phases are mature
- update this roadmap whenever a phase materially changes status
- note for each completed phase:
  - what data became available
  - what new risk was reduced
  - what the next phase now depends on

Current live trading path notes:

- Orders are submitted from `custom/btc_agent/executor.py` through the upstream client in `agents/polymarket/polymarket.py`, which now uses the Polymarket CLOB V2 Python SDK.
- Required approvals/allowances are not auto-initialized by the custom BTC flow; the wallet must already be approved for Polymarket / CTF execution.
- Proxy wallets are supported in the custom BTC live path through `POLYMKT_PROXY_ADDRESS`, which is passed to the CLOB client as the `funder` with `signature_type=2`.
- Live execution is gated by `USE_PAPER_TRADES=false`, the normal trade-quality checks, successful cash-balance verification, and a cash balance large enough to fund the configured share size at the recommended limit price plus the estimated maker fee.
- Live execution no longer embeds `feeRateBps` in signed orders; the current BTC agent still uses `BTC_AGENT_LIVE_FEE_RATE_BPS` only as a conservative cash-availability estimate before submission.
- Live execution also depends on the trade size being auto-scaled enough to meet `BTC_AGENT_LIVE_MIN_ORDER_USD`.

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
- Added a hard startup gate in `custom/btc_agent/main.py` that runs the public-IP check and aborts execution before any further BTC-agent logic when the detected location is not in the allowed-country set.
- Adjusted `scripts/python/check_public_ip_indonesia.py` type annotations to remain compatible with the repo’s Python 3.9 runtime after the Indonesia startup gate was added.
- Updated the BTC paper-trading loop to log the exact execution snapshot and to reuse a single decision-time quote snapshot end-to-end per tick, eliminating mismatches between the printed `UP/DOWN (with decision)` snapshot and the final paper execution result.
- Added `BTC_AGENT_TRADE_SHARES_SIZE` and `BTC_AGENT_MAX_TRADES_PER_PERIOD`, switched paper-trade sizing to fixed shares with a minimum of 5, and introduced per-period in-memory active paper-order tracking so the loop can stop decisioning after the trade limit is reached and report each active order’s BTC target plus winning/losing status until the next 5-minute window starts.
- Added `BTC_AGENT_DEBUG` with a default of `false` and changed the BTC loop so non-debug mode emits only compact operational output while debug mode retains the fuller diagnostic logs; account balances now print only on the first loop and at each new 5-minute market period.
- Updated BTC spot-price fetching to fail over across multiple live providers before reusing the most recent in-memory BTC price sample, reducing the chance that a single-provider rate limit such as CoinGecko HTTP 429 will crash active paper-order status checks after the process has already collected a recent BTC sample.
- Added `USE_PAPER_TRADES` with a default of `true` and wired the custom BTC executor to submit live Polymarket buy orders through the upstream `agents/polymarket/polymarket.py` client when set to `false`; the live path now aborts the process if the account cash balance cannot fund the configured trade size at the recommended limit price or if the cash balance cannot be verified safely, and it supports proxy-wallet execution by passing `POLYMKT_PROXY_ADDRESS` into the upstream CLOB client as a `signature_type=2` funder.
- Added `BTC_AGENT_LIVE_FEE_RATE_BPS` with a default of `1000` and updated the upstream Polymarket client wrapper plus the custom BTC live executor to pass that maker fee into live order submissions after the API rejected the default zero-fee live orders.
- Tightened the live-trading cash-balance guard to include the estimated maker fee in the required cash calculation before live order submission.
- Added `BTC_AGENT_LIVE_MIN_ORDER_USD` with a default of `1` and updated the custom BTC live executor to auto-scale live order size upward so the order notional meets the exchange’s minimum marketable buy amount.
- Corrected the BTC window-open approximation to use the earliest retained sample inside the current 5-minute market window instead of the oldest retained sample across the whole process history, preventing active-order targets from drifting across periods.
- Updated BTC feature readiness at 5-minute boundaries to carry the most recent pre-window BTC sample forward as the new window baseline, so the first tick of a new market period can reuse retained history instead of forcing an extra warmup tick.
- Updated BTC feature analysis so 5-minute momentum and volatility use the last 5 minutes of retained BTC samples regardless of the current market-window boundary, while preserving a separate market-window-open reference for threshold context.
- Clarified paper-order settlement fallback so active-order status uses the period-start BTC reference when an explicit Polymarket threshold is unavailable, instead of relying on the trade-entry BTC price.
- Added `price_to_beat` to the runtime market output and to the LLM prompt so each 5-minute decision is made with the market’s displayed baseline when available.

### 2026-04-22

- Added `AI_ENGINE`-based LLM selection so the BTC agent can route decisions through OpenAI or Gemini using the matching provider API key from `.env`.
- Updated `.env.example` to document the active BTC-agent AI-engine and model environment variables.
- Replaced the Gemini-specific timeout knobs with generic `API_CONNECTION_TIMEOUT`, `API_CONNECTION_RETRY_TIMER`, and `API_CONNECTION_RETRY_ATTEMPTS` controls so each LLM attempt is bounded, logged, and retried consistently before the BTC loop moves to the next tick.
- Expanded the startup geolocation allowlist so the public IP check now accepts Mexico in addition to Indonesia.
- Updated the launch path to export `.env` before startup and documented `HTTP_PROXY` / `HTTPS_PROXY` support so VPN or proxy-routed outbound traffic can be configured explicitly.
- Added explicit SOCKS proxy support for Mullvad-style `ALL_PROXY` usage by routing active BTC-agent requests through normalized proxy settings and adding the required SOCKS client dependencies.
- Added a pre-LLM quote gate so the BTC loop skips AI decision calls when both outcome tokens are already unsubmitable at current prices.
- Added `MINIMUM_WALLET_BALANCE` so the BTC loop aborts when available trading cash falls below the configured wallet floor.
- Added a last-minute market gate so the BTC loop stops making LLM decisions or trade attempts during the final 60 seconds of a market window.

### 2026-05-03

- Replaced the old variable share-budget sizing path with a fixed `SHARES_PER_TRADE` sizing model in `custom/btc_agent/config.py`, so both paper and live orders now use the same repo-defined share count per trade.
- Changed live execution so the agent no longer exceeds the configured budget to satisfy exchange minimum sizes; when the venue minimum cannot be met within budget, the attempt is rejected cleanly and the loop continues running.
- Added failed directional order-attempt logging under `completed_orders/completed_order_attempt_<slug_ts>.txt`, preserving the same market, feature, quote, and regime context plus the failure reason.
- Updated finalized completed-order filenames to include the order side, for example `completed_order_win_up_<slug_ts>.txt` and `completed_order_loss_down_<slug_ts>.txt`.
- Clarified that the code still contains the per-period trade-limit plumbing, but current operation is intentionally back on `BTC_AGENT_MAX_TRADES_PER_PERIOD=1` and future multi-trade-per-period behavior will be handled in a later phase.
