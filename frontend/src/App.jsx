import { useEffect, useState, useCallback, useRef, useMemo, createContext, lazy, Suspense } from "react";
import MapView from "./components/MapView.jsx";
import SidePanel from "./components/SidePanel.jsx";
import SearchBar from "./components/SearchBar.jsx";
import { findNearestTract } from "./utils/geo.js";
import { t } from "./i18n.js";

// Code-split the heavy / less-used tabs so the initial bundle stays small.
const SensorPlacement = lazy(() => import("./components/SensorPlacement.jsx"));
const AirQualityGuide = lazy(() => import("./components/AirQualityGuide.jsx"));

const preloadSensorPlacement = () => import("./components/SensorPlacement.jsx");
const preloadAirQualityGuide = () => import("./components/AirQualityGuide.jsx");

// API base URL: in production set VITE_API_URL=https://your-backend.onrender.com
// In dev, leave empty so /api/... uses the Vite proxy to localhost:8000
const API_BASE = import.meta.env.VITE_API_URL || "";
// The backend refreshes PurpleAir every PURPLEAIR_CACHE_TTL_MIN (15 min by
// default) and recomputes predictions right after, on its own schedule. Polling
// every 5 min shows each new batch within ~5 min; unchanged polls are cheap
// because the response carries an ETag and the browser revalidates with a 304.
const POLL_MS = 5 * 60 * 1000;
// Render free-tier cold starts take 30-60 s, so back off quickly at first and
// then settle at one attempt per minute.
const RETRY_DELAYS_MS = [2000, 5000, 15000, 30000, 60000];
const retryDelay = (attempt) => RETRY_DELAYS_MS[Math.min(attempt, RETRY_DELAYS_MS.length - 1)];
// Failures before the map overlay stops saying "waking up" and says "error".
const GEOJSON_WAKING_ATTEMPTS = 4;
// The first sensor-placement solve after a cold start can outlast the gateway.
const QUANTUM_RETRY_DELAYS_MS = [5000, 15000, 30000];
const VISIT_KEY = "ssi_visit_sent";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export const LanguageContext = createContext({ lang: "en", setLang: () => {} });

function TabFallback() {
  return (
    <div className="loading-wrap" aria-busy="true" aria-live="polite">
      <div className="sk-row">
        <div className="skeleton sk-h" style={{ width: "60%" }} />
        <div className="skeleton sk-h" style={{ width: "40%", height: 10 }} />
      </div>
      <div className="skeleton sk-card" style={{ marginTop: 14 }} />
      <div className="skeleton sk-card" />
      <div className="skeleton sk-card" />
    </div>
  );
}

export default function App() {
  const [predictions, setPredictions] = useState(null);
  const [geojson, setGeojson] = useState(null);
  const [geojsonStatus, setGeojsonStatus] = useState("loading");
  // Store only the geoid; the tract itself is derived from the latest
  // predictions so a background refresh updates the open tract too.
  const [selectedGeoid, setSelectedGeoid] = useState(null);
  const [localWeather, setLocalWeather] = useState(null); // { geoid, weather }
  const [weatherLoading, setWeatherLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("map");
  const [searchMarker, setSearchMarker] = useState(null);
  const [searchExpanded, setSearchExpanded] = useState(false);
  const mapRef = useRef(null);
  const [visitCount, setVisitCount] = useState(null);

  const predictionsRef = useRef(null);
  const predTimerRef = useRef(null);
  const predAttemptRef = useRef(0);
  const predInFlightRef = useRef(false);
  const lastFetchRef = useRef(0); // client time of the last successful poll
  const geoTimerRef = useRef(null);
  const geoAttemptRef = useRef(0);
  const geoInFlightRef = useRef(false);
  const geoLoadedRef = useRef(false);
  const weatherCtrlRef = useRef(null);
  const pendingSearchRef = useRef(null);
  const visitStateRef = useRef("idle");

  // Quantum sensor placement state
  const [quantumData, setQuantumData] = useState(null);
  const [quantumLoading, setQuantumLoading] = useState(false);
  const [quantumError, setQuantumError] = useState(null);
  const [quantumSensors, setQuantumSensors] = useState(null);
  const [existingSensors, setExistingSensors] = useState(null);

  // Language (persisted)
  const [lang, setLang] = useState(() => {
    try { return localStorage.getItem("ssi_lang") || "en"; } catch (e) { return "en"; }
  });
  const langCtx = useMemo(() => ({ lang, setLang }), [lang]);

  useEffect(() => {
    try { localStorage.setItem("ssi_lang", lang); } catch (e) {}
    // Screen readers and hyphenation follow the page language.
    document.documentElement.lang = lang;
  }, [lang]);

  useEffect(() => { predictionsRef.current = predictions; }, [predictions]);

  const fetchGeojson = useCallback(async () => {
    if (geoLoadedRef.current || geoInFlightRef.current) return;
    geoInFlightRef.current = true;
    clearTimeout(geoTimerRef.current);
    try {
      const res = await fetch(`${API_BASE}/api/texas/tracts/geojson`);
      if (!res.ok) throw new Error(`GeoJSON error ${res.status}`);
      const data = await res.json();
      if (!(data.features?.length > 0 && data.features[0].geometry)) {
        throw new Error("GeoJSON has no tract geometry");
      }
      geoLoadedRef.current = true;
      setGeojson(data);
      setGeojsonStatus("ready");
    } catch (e) {
      const attempt = geoAttemptRef.current++;
      console.warn(`Tract GeoJSON failed (attempt ${attempt + 1}), retrying:`, e.message);
      setGeojsonStatus(attempt + 1 < GEOJSON_WAKING_ATTEMPTS ? "retrying" : "error");
      geoTimerRef.current = setTimeout(fetchGeojson, retryDelay(attempt));
    } finally {
      geoInFlightRef.current = false;
    }
  }, []);

  const fetchPredictions = useCallback(async () => {
    if (predInFlightRef.current) return;
    predInFlightRef.current = true;
    clearTimeout(predTimerRef.current);
    setRefreshing(true);
    let ok = false;
    try {
      const res = await fetch(`${API_BASE}/api/texas/predictions`);
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data = await res.json();
      ok = true;
      lastFetchRef.current = Date.now();
      predAttemptRef.current = 0;
      // Most polls return the batch we already have; keeping the old object
      // avoids restyling every tract polygon for nothing.
      setPredictions((prev) => (prev && prev.generated_at === data.generated_at ? prev : data));
      setError(null);
      // A successful poll means the server is awake, so retry the tract
      // shapes now instead of waiting out the backoff.
      if (!geoLoadedRef.current) fetchGeojson();
    } catch (e) {
      console.error("Failed to fetch predictions:", e);
      predAttemptRef.current += 1;
      setError(e.message);
    } finally {
      predInFlightRef.current = false;
      setRefreshing(false);
    }
    // Hidden tabs don't poll; the visibilitychange handler resumes.
    if (!document.hidden) {
      predTimerRef.current = setTimeout(
        fetchPredictions,
        ok ? POLL_MS : retryDelay(predAttemptRef.current - 1),
      );
    }
  }, [fetchGeojson]);

  useEffect(() => {
    fetchPredictions();
    fetchGeojson();
    const onVisibility = () => {
      clearTimeout(predTimerRef.current);
      if (document.hidden) return;
      const age = Date.now() - lastFetchRef.current;
      if (!lastFetchRef.current || age >= POLL_MS) fetchPredictions();
      else predTimerRef.current = setTimeout(fetchPredictions, POLL_MS - age);
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      clearTimeout(predTimerRef.current);
      clearTimeout(geoTimerRef.current);
    };
  }, [fetchPredictions, fetchGeojson]);

  // Count a visit at most once per browser session; later loads in the same
  // session only read the total. Re-tried once predictions arrive, in case the
  // first attempt hit a sleeping server.
  const predictionsLoaded = predictions != null;
  useEffect(() => {
    if (visitStateRef.current !== "idle") return;
    visitStateRef.current = "inflight";
    let alreadySent = false;
    try { alreadySent = sessionStorage.getItem(VISIT_KEY) === "1"; } catch (e) {}
    (async () => {
      try {
        const res = alreadySent
          ? await fetch(`${API_BASE}/api/metrics`)
          : await fetch(`${API_BASE}/api/visit`, { method: "POST" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!alreadySent) {
          try { sessionStorage.setItem(VISIT_KEY, "1"); } catch (e) {}
        }
        if (typeof data?.visits === "number") setVisitCount(data.visits);
        visitStateRef.current = "done";
      } catch (e) {
        console.warn("Failed to record visit:", e.message);
        visitStateRef.current = "idle";
      }
    })();
  }, [predictionsLoaded]);

  const tractsByGeoid = useMemo(() => {
    const m = new Map();
    predictions?.tracts?.forEach((tr) => m.set(tr.geoid, tr));
    return m;
  }, [predictions]);
  const selectedTract = selectedGeoid ? tractsByGeoid.get(selectedGeoid) ?? null : null;

  const handleTractSelect = useCallback(async (geoid) => {
    if (!predictionsRef.current?.tracts?.some((tr) => tr.geoid === geoid)) return;

    // Only the newest request may write weather or clear the loading badge,
    // so a slow response for an earlier click can't land on this tract.
    weatherCtrlRef.current?.abort();
    const ctrl = new AbortController();
    weatherCtrlRef.current = ctrl;

    setSelectedGeoid(geoid);
    setLocalWeather(null);
    setWeatherLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/tract/${geoid}`, { signal: ctrl.signal });
      if (res.ok) {
        const data = await res.json();
        if (weatherCtrlRef.current === ctrl) setLocalWeather({ geoid, weather: data.weather });
      }
    } catch (e) {
      if (e.name !== "AbortError") console.error("Failed to fetch local weather:", e);
    } finally {
      if (weatherCtrlRef.current === ctrl) {
        weatherCtrlRef.current = null;
        setWeatherLoading(false);
      }
    }
  }, []);

  const handleDeselect = useCallback(() => {
    weatherCtrlRef.current?.abort();
    weatherCtrlRef.current = null;
    setSelectedGeoid(null);
    setLocalWeather(null);
    setWeatherLoading(false);
  }, []);

  const fetchQuantumData = useCallback(async () => {
    setQuantumLoading(true);
    setQuantumError(null);
    for (let attempt = 0; ; attempt++) {
      try {
        const res = await fetch(`${API_BASE}/api/quantum/sensor-placement`);
        if (!res.ok) {
          const err = new Error(`API error ${res.status}`);
          err.retryable = res.status >= 500;
          throw err;
        }
        const data = await res.json();
        setQuantumData(data);
        if (data?.methods?.quantum_annealing?.selected_tracts) {
          setQuantumSensors(data.methods.quantum_annealing.selected_tracts);
        }
        if (Array.isArray(data?.existing_sensors)) setExistingSensors(data.existing_sensors);
        break;
      } catch (e) {
        // Network errors (no .retryable) are usually the server waking up.
        if (e.retryable !== false && attempt < QUANTUM_RETRY_DELAYS_MS.length) {
          await sleep(QUANTUM_RETRY_DELAYS_MS[attempt]);
          continue;
        }
        console.error("Failed to fetch quantum data:", e);
        setQuantumError(e.message);
        break;
      }
    }
    setQuantumLoading(false);
  }, []);

  const handleTabChange = useCallback((tab) => {
    if (!document.startViewTransition) {
      setActiveTab(tab);
    } else {
      document.startViewTransition(() => {
        setActiveTab(tab);
      });
    }
    if (tab === "quantum" && !quantumData && !quantumLoading) fetchQuantumData();
  }, [quantumData, quantumLoading, fetchQuantumData]);

  const handleViewSensor = useCallback((sensor) => {
    if (sensor?.lat && sensor?.lon) {
      setSearchMarker({ lat: sensor.lat, lon: sensor.lon });
      handleTractSelect(sensor.geoid);
    }
  }, [handleTractSelect]);

  const selectNearestTract = useCallback((coords) => {
    const nearest = findNearestTract(coords.lat, coords.lon, predictionsRef.current?.tracts);
    if (nearest) handleTractSelect(nearest.geoid);
  }, [handleTractSelect]);

  // Returns "ok", or "pending" when predictions haven't loaded yet: the map
  // still flies to the place and the tract is selected once data arrives.
  const handleSearch = useCallback((coords) => {
    setSearchMarker({ lat: coords.lat, lon: coords.lon });
    if (!predictionsRef.current?.tracts?.length) {
      pendingSearchRef.current = coords;
      return "pending";
    }
    pendingSearchRef.current = null;
    selectNearestTract(coords);
    setSearchExpanded(false);
    return "ok";
  }, [selectNearestTract]);

  useEffect(() => {
    if (!predictions?.tracts?.length || !pendingSearchRef.current) return;
    const coords = pendingSearchRef.current;
    pendingSearchRef.current = null;
    selectNearestTract(coords);
  }, [predictions, selectNearestTract]);

  const localWeatherForTract =
    localWeather && localWeather.geoid === selectedGeoid ? localWeather.weather : null;

  const sensorMarkersForMap = activeTab === "quantum" ? quantumSensors : null;
  const existingSensorsForMap = activeTab === "quantum" ? existingSensors : null;

  return (
    <LanguageContext.Provider value={langCtx}>
      <div className="app-layout">
        <div className="sidebar">
          <div className="sidebar-tabs" role="tablist" aria-label={t(lang, "tabs.aria")}>
            <button
              role="tab"
              aria-selected={activeTab === "map"}
              className={`sidebar-tab${activeTab === "map" ? " active" : ""}`}
              onClick={() => handleTabChange("map")}
            >
              {t(lang, "tabs.map")}
            </button>
            <button
              role="tab"
              aria-selected={activeTab === "quantum"}
              className={`sidebar-tab${activeTab === "quantum" ? " active" : ""}`}
              onClick={() => handleTabChange("quantum")}
              onMouseEnter={preloadSensorPlacement}
            >
              {t(lang, "tabs.sensors")}
            </button>
            <button
              role="tab"
              aria-selected={activeTab === "guide"}
              className={`sidebar-tab${activeTab === "guide" ? " active" : ""}`}
              onClick={() => handleTabChange("guide")}
              onMouseEnter={preloadAirQualityGuide}
            >
              {t(lang, "tabs.guide")}
            </button>
          </div>

          {activeTab === "map" && (
            <SidePanel
              predictions={predictions}
              selectedTract={selectedTract}
              localWeather={localWeatherForTract}
              onDeselect={handleDeselect}
              refreshing={refreshing}
              weatherLoading={weatherLoading}
              error={error}
              statewide={true}
              visitCount={visitCount}
            />
          )}

          {activeTab === "quantum" && (
            <Suspense fallback={<TabFallback />}>
              <SensorPlacement
                quantumData={quantumData}
                loading={quantumLoading}
                error={quantumError}
                onRetry={fetchQuantumData}
                onViewSensor={handleViewSensor}
                selectedTract={selectedTract}
                onDeselect={handleDeselect}
                visitCount={visitCount}
              />
            </Suspense>
          )}

          {activeTab === "guide" && (
            <Suspense fallback={<TabFallback />}>
              <AirQualityGuide />
            </Suspense>
          )}
        </div>

        <div className="map-wrapper">
          <button
            type="button"
            className="mobile-search-toggle"
            aria-label={t(lang, searchExpanded ? "search.close" : "search.open")}
            aria-expanded={searchExpanded}
            onClick={() => setSearchExpanded((v) => !v)}
          >
            {searchExpanded ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="11" cy="11" r="7" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
            )}
          </button>
          <div className={`map-search-overlay${searchExpanded ? " expanded" : ""}`}>
            <SearchBar onSearch={handleSearch} predictionsReady={predictionsLoaded} />
          </div>
          <MapView
            ref={mapRef}
            geojson={geojson}
            geojsonStatus={geojsonStatus}
            predictions={predictions?.tracts ?? []}
            onTractSelect={handleTractSelect}
            onBackgroundClick={handleDeselect}
            selectedGeoid={selectedGeoid}
            searchMarker={searchMarker}
            statewide={true}
            sensorMarkers={sensorMarkersForMap}
            existingSensors={existingSensorsForMap}
            activeTab={activeTab}
          />
        </div>
      </div>
    </LanguageContext.Provider>
  );
}
