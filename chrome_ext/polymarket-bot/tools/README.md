# Polymarket BTC Agent Extension

Personal-use Chrome extension scaffold for `polymarket.com`.

Current scope:

- inject an in-page output window on Polymarket pages
- capture the current browser slug on a configured interval
- auto-navigate to the live BTC Up/Down 5-minute slug and roll forward each period
- keep a local config file for runtime, LLM, and trading settings

## Config

Edit [src/shared/config.js](/appl/agents/chrome_ext/polymarket-bot/src/shared/config.js) for:

- `runtime.snapshotIntervalMs`
- `llm.provider`
- `llm.openaiApiKey`
- `llm.geminiApiKey`
- future trading defaults

This config is bundled into the extension build. That is acceptable for the current personal-use workflow.

## Commands

```bash
npm run dev
npm run build
```

After `npm run build`, load the unpacked extension from [build](/appl/agents/chrome_ext/polymarket-bot/build).

## Current behavior

On `https://polymarket.com/*` and `https://www.polymarket.com/*`, the extension injects a fixed output panel into the page, redirects to the live `btc-updown-5m-<unix_timestamp>` market for the current 5-minute window, and records the current page slug every `snapshotIntervalMs`.

The sidebar is also wired, but the main runtime currently lives in the page DOM via the content script.
