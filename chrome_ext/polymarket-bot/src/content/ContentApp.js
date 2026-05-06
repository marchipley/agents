import {extensionConfig} from '../shared/config.js'
import {
  readCurrentBtcPrice,
  readOutcomeLimitPrices,
  readPeriodOpenPriceToBeat
} from './domData.js'
import {buildSlugSnapshot} from './slug.js'

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

  const snapshotOutput = document.createElement('pre')
  snapshotOutput.className = 'pm_agent_panel_snapshot'

  const logHeading = document.createElement('div')
  logHeading.className = 'pm_agent_panel_log_heading'
  logHeading.textContent = 'Snapshots'

  const logList = document.createElement('div')
  logList.className = 'pm_agent_panel_log'

  header.append(title, closeButton)
  container.append(header, meta, snapshotOutput, logHeading, logList)

  const snapshots = []
  const marketCache = new Map()
  let activePathname = window.location.pathname
  let intervalId = null
  let isClosed = false
  let dragState = null
  let lastNavigationPath = null

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
        ` | Next period in ${latestSnapshot.secondsUntilNextPeriod}s`
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
