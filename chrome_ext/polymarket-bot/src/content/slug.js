const BTC_UPDOWN_PREFIX = 'btc-updown-5m-'
const PERIOD_SECONDS = 300

export function getCurrentSlug(locationLike = window.location) {
  const pathname = locationLike.pathname || '/'
  const segments = pathname.split('/').filter(Boolean)

  if (segments.length === 0) {
    return {
      pathname,
      slug: 'home'
    }
  }

  const eventIndex = segments.findIndex((segment) => segment === 'event')
  if (eventIndex >= 0 && segments[eventIndex + 1]) {
    return {
      pathname,
      slug: segments[eventIndex + 1]
    }
  }

  return {
    pathname,
    slug: segments[segments.length - 1]
  }
}

export function getCurrentPeriodStartTs(now = new Date()) {
  const nowTs = Math.floor(now.getTime() / 1000)
  return nowTs - (nowTs % PERIOD_SECONDS)
}

export function buildLiveBtcUpDownSlug(now = new Date()) {
  return `${BTC_UPDOWN_PREFIX}${getCurrentPeriodStartTs(now)}`
}

export function buildLiveBtcUpDownPath(now = new Date()) {
  return `/event/${buildLiveBtcUpDownSlug(now)}`
}

export function getNextPeriodStartTs(now = new Date()) {
  return getCurrentPeriodStartTs(now) + PERIOD_SECONDS
}

export function getSecondsUntilNextPeriod(now = new Date()) {
  return Math.max(getNextPeriodStartTs(now) - Math.floor(now.getTime() / 1000), 0)
}

export function buildNextBtcUpDownSlug(now = new Date()) {
  return `${BTC_UPDOWN_PREFIX}${getNextPeriodStartTs(now)}`
}

export function buildSlugSnapshot(now = new Date()) {
  const {pathname, slug} = getCurrentSlug()
  const liveSlug = buildLiveBtcUpDownSlug(now)
  const nextSlug = buildNextBtcUpDownSlug(now)
  const currentPeriodStartTs = getCurrentPeriodStartTs(now)
  const nextPeriodStartTs = getNextPeriodStartTs(now)

  return {
    capturedAtIso: now.toISOString(),
    pathname,
    slug,
    liveSlug,
    nextSlug,
    currentPeriodStartTs,
    nextPeriodStartTs,
    secondsUntilNextPeriod: getSecondsUntilNextPeriod(now),
    isOnLiveSlug: slug === liveSlug
  }
}
