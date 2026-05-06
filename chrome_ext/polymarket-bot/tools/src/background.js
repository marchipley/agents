console.log('Polymarket BTC Agent background ready')

const isFirefoxLike =
  import.meta.env.EXTENSION_PUBLIC_BROWSER === 'firefox' ||
  import.meta.env.EXTENSION_PUBLIC_BROWSER === 'gecko-based'
const PERIOD_SECONDS = 300
const LIVE_MARKET_BASE_URL = 'https://polymarket.com/event/'

function disableSidePanelActionClick() {
  if (!chrome.sidePanel || !chrome.sidePanel.setPanelBehavior) return

  try {
    chrome.sidePanel.setPanelBehavior({openPanelOnActionClick: false})
  } catch (error) {
    console.error('Failed to disable side panel action click', error)
  }
}

function getLiveMarketUrl(now = new Date()) {
  const nowTs = Math.floor(now.getTime() / 1000)
  const windowStartTs = nowTs - (nowTs % PERIOD_SECONDS)
  return `${LIVE_MARKET_BASE_URL}btc-updown-5m-${windowStartTs}`
}

function openLiveMarketInActiveTab() {
  const targetUrl = getLiveMarketUrl()

  chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
    const activeTab = tabs && tabs[0]
    const activeTabId = activeTab && activeTab.id
    if (!activeTabId) return

    chrome.tabs.update(activeTabId, {url: targetUrl})
  })
}

disableSidePanelActionClick()

if (chrome.runtime.onInstalled) {
  chrome.runtime.onInstalled.addListener(() => {
    disableSidePanelActionClick()
  })
}

if (chrome.runtime.onStartup) {
  chrome.runtime.onStartup.addListener(() => {
    disableSidePanelActionClick()
  })
}

if (isFirefoxLike) {
  browser.browserAction.onClicked.addListener(() => {
    browser.tabs.query({active: true, currentWindow: true}).then((tabs) => {
      const activeTab = tabs && tabs[0]
      if (!activeTab || typeof activeTab.id !== 'number') return

      browser.tabs.update(activeTab.id, {url: getLiveMarketUrl()})
    })
  })

  browser.runtime.onMessage.addListener((message) => {
    if (!message || message.type !== 'openSidebar') return

    browser.sidebarAction.open()
  })
}

if (!isFirefoxLike) {
  chrome.action.onClicked.addListener(() => {
    disableSidePanelActionClick()
    openLiveMarketInActiveTab()
  })
}
