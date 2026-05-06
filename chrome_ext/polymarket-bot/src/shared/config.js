export const extensionConfig = {
  site: {
    allowedHosts: ['polymarket.com', 'www.polymarket.com']
  },
  runtime: {
    snapshotIntervalMs: 5000,
    maxSnapshots: 25,
    panelTitle: 'Polymarket BTC Agent',
    autoNavigateToLiveSlug: true
  },
  llm: {
    provider: 'OPENAI',
    openaiApiKey: '',
    openaiModel: 'gpt-4.1-mini',
    geminiApiKey: '',
    geminiModel: 'gemini-2.5-flash'
  },
  trading: {
    enabled: false,
    usePaperTrades: true,
    maxTradeUsd: 5
  }
}

export function isAllowedHost(hostname = window.location.hostname) {
  return extensionConfig.site.allowedHosts.includes(hostname)
}
