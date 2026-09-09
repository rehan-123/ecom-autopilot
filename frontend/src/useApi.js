import { useEffect, useState, useCallback } from 'react'
import { getJson } from './api'

// One hook for every screen: loading, error and data, with a manual refresh.
// Errors surface rather than rendering an empty table that looks like "no
// results" when it actually means "the backend is down".
export function useApi(path) {
  const [state, setState] = useState({ data: null, error: null, loading: true })

  const load = useCallback(() => {
    let cancelled = false
    setState((s) => ({ ...s, loading: true }))
    getJson(path)
      .then((data) => !cancelled && setState({ data, error: null, loading: false }))
      .catch((error) => !cancelled && setState({ data: null, error, loading: false }))
    return () => { cancelled = true }
  }, [path])

  useEffect(() => load(), [load])
  return { ...state, reload: load }
}
