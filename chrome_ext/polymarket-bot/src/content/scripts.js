import createContentApp from './ContentApp.js'
import './styles.css'
import {isAllowedHost} from '../shared/config.js'

export default function initial() {
  if (!isAllowedHost()) {
    return () => {}
  }

  const existingRoot = document.querySelector('[data-extension-root="true"]')
  if (existingRoot) {
    return () => {}
  }

  const rootDiv = document.createElement('div')
  rootDiv.setAttribute('data-extension-root', 'true')
  document.body.appendChild(rootDiv)

  const shadowRoot = rootDiv.attachShadow({mode: 'open'})
  const styleElement = document.createElement('style')
  shadowRoot.appendChild(styleElement)

  fetchCSS().then((response) => {
    styleElement.textContent = response
  })

  const container = createContentApp()
  shadowRoot.appendChild(container)

  return () => {
    if (typeof container.cleanup === 'function') {
      container.cleanup()
    }
    rootDiv.remove()
  }
}

async function fetchCSS() {
  const cssUrl = new URL('./styles.css', import.meta.url)
  const response = await fetch(cssUrl)
  const text = await response.text()
  return response.ok ? text : Promise.reject(text)
}
