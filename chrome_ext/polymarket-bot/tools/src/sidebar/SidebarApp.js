import './styles.css'
import {extensionConfig} from '../shared/config.js'

function SidebarApp() {
  const root = document.getElementById('root')
  if (!root) return

  root.innerHTML = `
    <div class="sidebar_app">
      <div class="sidebar_eyebrow">Polymarket only</div>
      <h1 class="sidebar_title">${extensionConfig.runtime.panelTitle}</h1>
      <p class="sidebar_description">
        The active monitor runs inside <code>polymarket.com</code> pages and
        captures the current browser slug on a configurable interval while
        auto-navigating to the live <code>btc-updown-5m-&lt;timestamp&gt;</code>
        market.
      </p>
      <div class="sidebar_card">
        <div class="sidebar_card_label">Snapshot interval</div>
        <code>${extensionConfig.runtime.snapshotIntervalMs}ms</code>
      </div>
      <div class="sidebar_card">
        <div class="sidebar_card_label">LLM provider</div>
        <code>${extensionConfig.llm.provider}</code>
      </div>
      <div class="sidebar_card">
        <div class="sidebar_card_label">Trading enabled</div>
        <code>${String(extensionConfig.trading.enabled)}</code>
      </div>
      <p class="sidebar_note">
        Edit <code>src/shared/config.js</code> to set your local keys and runtime
        defaults for the next iteration.
      </p>
    </div>
  `
}

SidebarApp()
