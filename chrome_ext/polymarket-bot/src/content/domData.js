function normalizeDecimalText(text) {
  if (!text) return null

  const normalized = String(text).replace(/[$,\s]/g, '')
  const value = Number(normalized)
  if (!Number.isFinite(value)) return null

  return normalized
}

function findPriceToBeatElement() {
  return Array.from(document.querySelectorAll('span')).find((element) => {
    const classList = element.classList
    return (
      classList.contains('text-text-secondary') &&
      classList.contains('text-heading-2xl')
    )
  })
}

function getDigitValue(element) {
  const currentValue = element.style.getPropertyValue('--current').trim()
  if (!/^\d$/.test(currentValue)) return null
  return currentValue
}

function findBtcPriceHost() {
  return Array.from(document.querySelectorAll('div')).find((element) => {
    const classList = element.classList
    return (
      classList.contains('flex') &&
      classList.contains('items-center') &&
      classList.contains('gap-1') &&
      Array.from(classList).some((className) => className.includes('font-[620]'))
    )
  })
}

function normalizeCentText(text) {
  if (!text) return null

  const match = String(text).match(/([0-9]+(?:\.[0-9]+)?)\s*¢/)
  if (!match) return null

  const centsValue = Number(match[1])
  if (!Number.isFinite(centsValue)) return null

  return (centsValue / 100).toFixed(2)
}

export function readPeriodOpenPriceToBeat() {
  const element = findPriceToBeatElement()
  if (!element) return null
  return normalizeDecimalText(element.textContent)
}

export function readCurrentBtcPrice() {
  const hostContainer = findBtcPriceHost()
  const numberFlow = hostContainer && hostContainer.querySelector('number-flow-react')
  const shadowRoot = numberFlow && numberFlow.shadowRoot
  if (!shadowRoot) return null

  const integerDigits = Array.from(
    shadowRoot.querySelectorAll('[part~="integer-digit"]')
  )
    .map(getDigitValue)
    .filter(Boolean)
    .join('')

  const fractionDigits = Array.from(
    shadowRoot.querySelectorAll('[part~="fraction-digit"]')
  )
    .map(getDigitValue)
    .filter(Boolean)
    .join('')

  if (!integerDigits) return null

  const priceText = fractionDigits
    ? `${integerDigits}.${fractionDigits}`
    : integerDigits

  const parsedPrice = Number(priceText)
  if (!Number.isFinite(parsedPrice)) return null

  return parsedPrice.toFixed(2)
}

export function readOutcomeLimitPrices() {
  const buttons = Array.from(
    document.querySelectorAll('button.trading-button[data-color]')
  )

  const upButton = buttons.find(
    (button) => button.getAttribute('data-color') === 'green'
  )
  const downButton = buttons.find(
    (button) => button.getAttribute('data-color') === 'gray'
  )

  return {
    upLimitPrice: normalizeCentText(upButton && upButton.textContent),
    downLimitPrice: normalizeCentText(downButton && downButton.textContent)
  }
}
