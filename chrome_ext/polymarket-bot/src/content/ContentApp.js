import {extensionConfig} from '../shared/config.js'
import {
  readCurrentBtcPrice,
  readOutcomeLimitPrices,
  readPeriodOpenPriceToBeat
} from './domData.js'
import {buildSlugSnapshot} from './slug.js'

const AUTO_REFRESH_STORAGE_KEY = 'pm-agent-auto-refresh-enabled'

export default function createContentApp() {
  const container = document.createElement('section')
  container.className = 'pm_agent_panel'
  container.style.right = '16px'
  container.style.bottom = '16px'

  const header = document.createElement('div')
  header.className = 'pm_agent_panel_header'

  const title = document.createElement('div')
  title.className = 'pm_agent_panel_title'
  title.textContent = extensionConfig.runtime.panelTitle

  const closeButton = document.createElement('button')
  closeButton.type = 'button'
  closeButton.className = 'pm_agent_panel_close'
  closeButton.setAttribute('aria-label', 'Close panel')
  closeButton.textContent = 'x'

  const meta = document.createElement('div')
  meta.className = 'pm_agent_panel_meta'
  meta.textContent = `Interval ${extensionConfig.runtime.snapshotIntervalMs}ms`

  const controls = document.createElement('div')
  controls.className = 'pm_agent_panel_controls'

  const controlsLabel = document.createElement('span')
  controlsLabel.className = 'pm_agent_panel_controls_label'
  controlsLabel.textContent = 'Auto-refresh'

  const refreshToggle = document.createElement('button')
  refreshToggle.type = 'button'
  refreshToggle.className = 'pm_agent_panel_toggle'
  refreshToggle.setAttribute('aria-label', 'Toggle auto-refresh')

  const snapshotOutput = document.createElement('pre')
  snapshotOutput.className = 'pm_agent_panel_snapshot'

  const logHeading = document.createElement('div')
  logHeading.className = 'pm_agent_panel_log_heading'
  logHeading.textContent = 'Snapshots'

  const logList = document.createElement('div')
  logList.className = 'pm_agent_panel_log'

  controls.append(controlsLabel, refreshToggle)
  header.append(title, closeButton)
  container.append(header, meta, controls, snapshotOutput, logHeading, logList)

  const snapshots = []
  const marketCache = new Map()
  let activePathname = window.location.pathname
  let intervalId = null
  let isClosed = false
  let dragState = null
  let lastNavigationPath = null
  let autoRefreshEnabled = readAutoRefreshSetting()

  const formatSnapshotText = (snapshot) => {
    const periodOpenPriceToBeat = snapshot.periodOpenPriceToBeat || 'pending'
    const btcPrice = snapshot.btcPrice || 'pending'
    const upLimitPrice = snapshot.upLimitPrice || 'pending'
    const downLimitPrice = snapshot.downLimitPrice || 'pending'

    return [
      'Market:',
      `    slug                  = ${snapshot.slug}`,
      `    period_open_price_to_beat = ${periodOpenPriceToBeat}`,
      'Features:',
      `  btc_price             = ${btcPrice}`,
      `  up_limit_price        = ${upLimitPrice}`,
      `  down_limit_price      = ${downLimitPrice}`
    ].join('\n')
  }

  const render = (latestSnapshot) => {
    if (isClosed) return

    if (latestSnapshot) {
      snapshotOutput.textContent = formatSnapshotText(latestSnapshot)
      meta.textContent =
        `Interval ${extensionConfig.runtime.snapshotIntervalMs}ms` +
        ` | Next period in ${latestSnapshot.secondsUntilNextPeriod}s` +
        (autoRefreshEnabled ? '' : ' | Refresh paused')
    }

    logList.replaceChildren(
      ...snapshots.map((snapshot) => {
        const item = document.createElement('pre')
        item.className = 'pm_agent_panel_log_item'
        item.textContent =
          `${snapshot.capturedAtIso}\n${formatSnapshotText(snapshot)}`
        return item
      })
    )
  }

  const renderPausedMeta = (snapshot) => {
    meta.textContent =
      `Interval ${extensionConfig.runtime.snapshotIntervalMs}ms` +
      ` | Next period in ${snapshot.secondsUntilNextPeriod}s | Refresh paused`
  }

  const updateRefreshToggle = () => {
    refreshToggle.textContent = autoRefreshEnabled ? 'On' : 'Off'
    refreshToggle.setAttribute('aria-pressed', String(autoRefreshEnabled))
  }

  const navigateToLiveMarketIfNeeded = (snapshot) => {
    if (isClosed || !extensionConfig.runtime.autoNavigateToLiveSlug) {
      return
    }

    const targetPath = `/event/${snapshot.liveSlug}`
    if (window.location.pathname === targetPath) {
      lastNavigationPath = null
      return
    }

    if (lastNavigationPath === targetPath) {
      return
    }

    lastNavigationPath = targetPath
    window.location.replace(targetPath)
  }

  const hydrateMarketSnapshot = (snapshot) => {
    const cachedMarket = marketCache.get(snapshot.slug)
    if (cachedMarket) {
      return {
        ...snapshot,
        periodOpenPriceToBeat: cachedMarket.periodOpenPriceToBeat
      }
    }

    const periodOpenPriceToBeat = readPeriodOpenPriceToBeat()
    if (periodOpenPriceToBeat) {
      marketCache.set(snapshot.slug, {periodOpenPriceToBeat})
    }

    return {
      ...snapshot,
      periodOpenPriceToBeat: periodOpenPriceToBeat || null
    }
  }

  const captureSnapshot = () => {
    if (isClosed) return

    let snapshot = buildSlugSnapshot()
    navigateToLiveMarketIfNeeded(snapshot)

    if (snapshot.pathname !== activePathname) {
      activePathname = snapshot.pathname
      snapshots.length = 0
    }

    if (!autoRefreshEnabled) {
      renderPausedMeta(snapshot)
      return
    }

    snapshot = hydrateMarketSnapshot(snapshot)
    const {upLimitPrice, downLimitPrice} = readOutcomeLimitPrices()
    snapshot = {
      ...snapshot,
      btcPrice: readCurrentBtcPrice(),
      upLimitPrice,
      downLimitPrice
    }

    snapshots.unshift(snapshot)
    snapshots.splice(extensionConfig.runtime.maxSnapshots)
    render(snapshot)
  }

  const closePanel = () => {
    if (isClosed) return
    isClosed = true

    if (intervalId !== null) {
      window.clearInterval(intervalId)
      intervalId = null
    }

    container.remove()
  }

  const applyDragPosition = (clientX, clientY) => {
    if (!dragState) return

    const nextLeft = clientX - dragState.offsetX
    const nextTop = clientY - dragState.offsetY
    container.style.left = `${Math.max(8, nextLeft)}px`
    container.style.top = `${Math.max(8, nextTop)}px`
    container.style.right = 'auto'
    container.style.bottom = 'auto'
  }

  const handlePointerMove = (event) => {
    applyDragPosition(event.clientX, event.clientY)
  }

  const stopDragging = () => {
    dragState = null
    window.removeEventListener('pointermove', handlePointerMove)
    window.removeEventListener('pointerup', stopDragging)
  }

  header.addEventListener('pointerdown', (event) => {
    if (event.target === closeButton) return

    const rect = container.getBoundingClientRect()
    dragState = {
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', stopDragging)
  })

  closeButton.addEventListener('click', closePanel)
  refreshToggle.addEventListener('click', () => {
    autoRefreshEnabled = !autoRefreshEnabled
    writeAutoRefreshSetting(autoRefreshEnabled)
    updateRefreshToggle()

    if (autoRefreshEnabled) {
      captureSnapshot()
    } else {
      renderPausedMeta(buildSlugSnapshot())
    }
  })

  updateRefreshToggle()
  captureSnapshot()
  intervalId = window.setInterval(
    captureSnapshot,
    extensionConfig.runtime.snapshotIntervalMs
  )

  container.cleanup = () => {
    stopDragging()
    closePanel()
  }

  return container
}

function readAutoRefreshSetting() {
  try {
    const stored = window.localStorage.getItem(AUTO_REFRESH_STORAGE_KEY)
    if (stored === 'true') return true
    if (stored === 'false') return false
  } catch (error) {}

  return extensionConfig.runtime.snapshotsEnabledByDefault
}

function writeAutoRefreshSetting(enabled) {
  try {
    window.localStorage.setItem(
      AUTO_REFRESH_STORAGE_KEY,
      enabled ? 'true' : 'false'
    )
  } catch (error) {}
}
