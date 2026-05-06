(function () {
  const STORAGE_KEY = 'pm-agent-auto-refresh-enabled'
  const PANEL_SELECTOR = '[data-extension-root="true"]'
  const SNAPSHOT_SELECTOR = '.pm_agent_panel_snapshot'
  const LOG_SELECTOR = '.pm_agent_panel_log'

  function readEnabledState() {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY)
      return stored !== 'false'
    } catch (error) {
      return true
    }
  }

  function writeEnabledState(enabled) {
    try {
      window.localStorage.setItem(STORAGE_KEY, enabled ? 'true' : 'false')
    } catch (error) {}
  }

  function createToggleRow(button, enabled) {
    const row = document.createElement('div')
    row.setAttribute('data-pm-agent-refresh-toggle', 'true')
    row.style.cssText =
      'display:flex;align-items:center;justify-content:space-between;gap:12px;' +
      'margin:0 16px 12px;padding:10px 12px;border:1px solid rgba(123,224,255,.12);' +
      'border-radius:12px;background:rgba(255,255,255,.04);font-size:12px;line-height:1.4;'

    const label = document.createElement('span')
    label.textContent = 'Auto-refresh'
    label.style.cssText = 'color:rgba(232,246,255,.82);'

    button.type = 'button'
    button.style.cssText =
      'appearance:none;border:1px solid rgba(123,224,255,.2);' +
      'background:rgba(255,255,255,.06);color:#e8f6ff;padding:6px 10px;' +
      'border-radius:9999px;cursor:pointer;font:inherit;line-height:1;'
    button.textContent = enabled ? 'On' : 'Off'

    row.append(label, button)
    return row
  }

  function attachToggle(shadowRoot) {
    const panel = shadowRoot.querySelector('.pm_agent_panel')
    const snapshot = shadowRoot.querySelector(SNAPSHOT_SELECTOR)
    const log = shadowRoot.querySelector(LOG_SELECTOR)
    if (!panel || !snapshot || !log) return false
    if (shadowRoot.querySelector('[data-pm-agent-refresh-toggle="true"]')) return true

    let enabled = readEnabledState()
    let restoring = false
    let frozenSnapshotText = snapshot.textContent
    let frozenLogHtml = log.innerHTML

    const button = document.createElement('button')
    const row = createToggleRow(button, enabled)
    panel.insertBefore(row, snapshot)

    const freezeCurrentView = () => {
      frozenSnapshotText = snapshot.textContent
      frozenLogHtml = log.innerHTML
    }

    const updateButton = () => {
      button.textContent = enabled ? 'On' : 'Off'
    }

    const restoreFrozenView = () => {
      if (enabled || restoring) return
      restoring = true
      snapshot.textContent = frozenSnapshotText
      log.innerHTML = frozenLogHtml
      restoring = false
    }

    const observer = new MutationObserver(() => {
      if (enabled) {
        freezeCurrentView()
        return
      }
      restoreFrozenView()
    })

    observer.observe(snapshot, {
      childList: true,
      characterData: true,
      subtree: true
    })

    observer.observe(log, {
      childList: true,
      characterData: true,
      subtree: true
    })

    button.addEventListener('click', () => {
      if (enabled) {
        freezeCurrentView()
      }

      enabled = !enabled
      writeEnabledState(enabled)
      updateButton()

      if (enabled) {
        frozenSnapshotText = snapshot.textContent
        frozenLogHtml = log.innerHTML
      } else {
        restoreFrozenView()
      }
    })

    updateButton()
    if (!enabled) {
      restoreFrozenView()
    }

    return true
  }

  function boot() {
    const tries = 120
    let attempt = 0

    const timer = window.setInterval(() => {
      attempt += 1
      const root = document.querySelector(PANEL_SELECTOR)
      const shadowRoot = root && root.shadowRoot
      if (shadowRoot && attachToggle(shadowRoot)) {
        window.clearInterval(timer)
        return
      }

      if (attempt >= tries) {
        window.clearInterval(timer)
      }
    }, 250)
  }

  boot()
})()
