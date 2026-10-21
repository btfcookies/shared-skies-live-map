import { memo, useMemo, useRef, useEffect, useState, forwardRef, useCallback, useContext } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap, Circle, CircleMarker, Tooltip, Pane } from "react-leaflet";
import L from "leaflet";
import { BREAKPOINTS, pm25ToEpaAqi } from "../utils/aqi.js";
import { LanguageContext } from '../App';
import { t, translateCategory } from '../i18n';
import "./MapView.css";

// Basemap: Stadia Maps "Alidade Smooth Dark" -- the near-black successor to
// CARTO Dark Matter. CARTO began watermarking this site's tiles once traffic
// crossed its anonymous free quota, on every CARTO host, so no keyless CARTO
// endpoint works for this domain any more. Stadia's free tier covers this
// traffic with no watermark, and it authenticates by DOMAIN (the site's origins
// are registered in the Stadia dashboard), so no key or token appears in this
// bundle. Served over HTTP/2, so one host multiplexes tiles without the
// head-of-line queueing that made the single-host Esri fallback lag.
// This is a single combined layer (base + labels); Stadia does not split
// alidade into label-free and labels-only raster variants the way CARTO did.
//
// Fallback: Stadia answers 401 on origins that are not registered (every
// Vercel preview URL) and 429 when rate-limited, which leaves a black map.
// After a streak of tile errors with no successful tile in between we switch,
// once, to keyless Esri Dark Gray. It is slower and label-free, but readable.
const BASEMAPS = {
  stadia: {
    url: "https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png",
    attribution:
      '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxNativeZoom: 20,
  },
  esri: {
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attribution:
      'Powered by <a href="https://www.esri.com/">Esri</a> &mdash; Esri, HERE, Garmin, &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, and the GIS user community',
    // Dark Gray Base has real imagery only to z16; Leaflet upscales beyond.
    maxNativeZoom: 16,
  },
};
const TILE_ERROR_STREAK_FALLBACK = 6;

const TEXAS_CENTER = [31.5, -99.0];
const TEXAS_ZOOM = 6;

// API base — matches App.jsx convention.
const API_BASE = import.meta.env.VITE_API_URL || "";

// NOAA HMS smoke polygon styling by density tier.
const HMS_STYLE = {
  Light:   { fillColor: "#ffeb99", color: "#caa84a", fillOpacity: 0.32, weight: 1 },
  Medium:  { fillColor: "#ff9933", color: "#cc6600", fillOpacity: 0.42, weight: 1.2 },
  Heavy:   { fillColor: "#cc3300", color: "#80190f", fillOpacity: 0.55, weight: 1.4 },
  Unknown: { fillColor: "#888888", color: "#555555", fillOpacity: 0.30, weight: 1 },
};
const hmsStyle = (feat) => HMS_STYLE[feat?.properties?.density || "Unknown"] || HMS_STYLE.Unknown;
// Refresh HMS once an hour to match the cache TTL.
const HMS_REFRESH_MS = 60 * 60 * 1000;

// Stable pathOptions objects: react-leaflet calls setStyle whenever the
// pathOptions identity changes, so inline literals restyle every marker on
// every render.
const SEARCH_MARKER_STYLE = { color: "#fff", weight: 2, fillColor: "#0077b6", fillOpacity: 1 };
const EXISTING_SENSOR_STYLE = {
  color: "rgba(255,255,255,0.3)",
  weight: 1,
  fillColor: "rgba(255,255,255,0.25)",
  fillOpacity: 0.7,
};
const QUANTUM_SENSOR_STYLE = {
  color: "#ffffff", weight: 2, fillColor: "#38bdf8", fillOpacity: 0.96, className: "ssi-sensor",
};
const QUANTUM_TOP3_STYLE = {
  color: "#ffffff", weight: 2.5, fillColor: "#fbbf24", fillOpacity: 0.96, className: "ssi-sensor-top",
};

function normGeoid(val) {
  if (!val) return "";
  return String(val).padStart(11, "0");
}

// Resting choropleth style for one tract.
function tractStyle(pred, isSelected) {
  return {
    fillColor: pred ? pred.color : "#2d3436",
    // 0.70 (was 0.80): Stadia alidade ships a single combined base+labels
    // raster with no labels-only variant to draw on top, so the choropleth
    // buries Texas city names. Easing the resting fill lets the base labels
    // read through the light Good/Moderate tracts without muddying the color.
    fillOpacity: isSelected ? 0.95 : 0.70,
    color: isSelected ? "#ffffff" : (pred ? pred.color : "rgba(0,0,0,0.08)"),
    weight: isSelected ? 2.5 : 1.0,
    lineCap: "round",
    lineJoin: "round",
  };
}

function hoverStyle(isSelected) {
  return {
    fillOpacity: 0.92,
    weight: isSelected ? 2.5 : 1.8,
    color: isSelected ? "#ffffff" : "rgba(255,255,255,0.8)",
  };
}

// Keeps the legend and its toggle from acting as map clicks/scrolls: they are
// rendered inside the map container, so without this a tap on them reaches
// BackgroundClickHandler and clears the selected tract.
function isolateFromMap(el) {
  if (!el) return;
  L.DomEvent.disableClickPropagation(el);
  L.DomEvent.disableScrollPropagation(el);
}

function todayUtcIso() {
  return new Date().toISOString().slice(0, 10);
}

function FlyToHandler({ target }) {
  const map = useMap();
  const lastTargetRef = useRef(null);
  useEffect(() => {
    if (!target || isNaN(target.lat) || isNaN(target.lon)) return;
    const last = lastTargetRef.current;
    if (last && last.lat === target.lat && last.lon === target.lon) return;
    lastTargetRef.current = { lat: target.lat, lon: target.lon };
    map.flyTo([target.lat, target.lon], 13, { duration: 1.8, easeLinearity: 0.2 });
  }, [target, map]);
  return null;
}

function BackgroundClickHandler({ onBackgroundClick, justClickedRef }) {
  const map = useMap();
  useEffect(() => {
    const handleMapClick = () => {
      if (justClickedRef.current) {
        justClickedRef.current = false;
        return;
      }
      onBackgroundClick?.();
    };
    map.on("click", handleMapClick);
    return () => map.off("click", handleMapClick);
  }, [map, onBackgroundClick, justClickedRef]);
  return null;
}

// Input-adaptive wheel zoom (replaces BOTH the old flyTo-based handler and
// Leaflet's native one):
//  - Leaflet's native handler divides wheel deltas by 2x devicePixelRatio on
//    Windows and squashes them through a sigmoid — on scaled-display laptops
//    a trackpad's small deltas collapse to ~0 zoom ("barely zooms"), until a
//    momentum burst punches through ("randomly starts zooming"). This handler
//    uses RAW pixel deltas: linear, device-independent.
//  - Mouse notches (large discrete deltas) -> one ANIMATED step per notch,
//    cursor-anchored (same feel as a desktop mouse today).
//  - Trackpad streams / ctrl+pinch (small rapid deltas) -> immediate zoom,
//    batched to one update per animation frame — smooth continuous zooming.
//  - No Math.round (no snap-back no-ops), no flyTo (no zoom-out arc flash).
function AdaptiveWheelZoom() {
  const map = useMap();
  useEffect(() => {
    map.scrollWheelZoom.disable();
    const el = map.getContainer();
    const PX_PER_LEVEL = 115;  // ~one mouse notch (100-120px) = ~1 zoom level
    const NOTCH_PX = 50;       // deltas >= this are treated as mouse notches

    let animTarget = null;     // zoom target while an animated step is in flight
    let pendingDz = 0;         // trackpad delta accumulated for the next frame
    let pendingPt = null;
    let rafId = null;

    const clampZoom = (z) =>
      Math.max(map.getMinZoom(), Math.min(map.getMaxZoom(), z));

    const onZoomEnd = () => { animTarget = null; };

    const applyPending = () => {
      rafId = null;
      if (!pendingDz || !pendingPt) { pendingDz = 0; return; }
      const tz = clampZoom(map.getZoom() + pendingDz);
      const pt = pendingPt;
      pendingDz = 0;
      pendingPt = null;
      map.setZoomAround(pt, tz, { animate: false });
    };

    const onWheel = (e) => {
      e.preventDefault();
      let dy = e.deltaY;
      if (e.deltaMode === 1) dy *= 33;        // lines -> px (Firefox mice)
      else if (e.deltaMode === 2) dy *= 100;  // pages -> px
      if (!dy) return;
      let dz = Math.max(-1.5, Math.min(1.5, -dy / PX_PER_LEVEL));
      const pt = map.mouseEventToContainerPoint(e);

      if (Math.abs(dy) >= NOTCH_PX && !e.ctrlKey) {
        // Mouse notch: animated cursor-anchored step. Consecutive notches
        // build on the in-flight TARGET (getZoom() reports the animation's
        // start zoom, not where it is heading).
        const base = animTarget != null && map._animatingZoom ? animTarget : map.getZoom();
        animTarget = clampZoom(base + dz);
        map.setZoomAround(pt, animTarget, { animate: true });
      } else {
        // Trackpad / pinch stream: apply continuously, one map update per
        // frame so the canvas never redraws faster than it can paint.
        pendingDz = Math.max(-1.5, Math.min(1.5, pendingDz + dz));
        pendingPt = pt;
        if (rafId == null) rafId = requestAnimationFrame(applyPending);
      }
    };

    map.on("zoomend", onZoomEnd);
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      el.removeEventListener("wheel", onWheel);
      map.off("zoomend", onZoomEnd);
      if (rafId != null) cancelAnimationFrame(rafId);
    };
  }, [map]);
  return null;
}

function MapLifecycle() {
  const map = useMap();
  useEffect(() => {
    if (!map) return;
    const onZoomStart = () => document.body.classList.add("disable-transitions");
    const onZoomEnd = () => {
      // Strip the no-transitions guard immediately so labels/polys snap to the new
      // zoom in the same frame the tiles arrive — no lingering "stale text" pause.
      document.body.classList.remove("disable-transitions");
    };
    map.on("zoomstart", onZoomStart);
    map.on("zoomend", onZoomEnd);
    return () => {
      map.off("zoomstart", onZoomStart);
      map.off("zoomend", onZoomEnd);
    };
  }, [map]);
  return null;
}

// Fires on every map move frame: re-projects last hover lat/lng → new container px
// so the tooltip follows the map during drag, and re-applies hover outline style
// so canvas redraws during pan don't wipe the highlight. Module scope (refs as
// props) so parent re-renders don't remount it and re-bind the listener.
function MapMoveHandler({ tooltipDomRef, lastMouseLatLngRef, hoveredLayerRef, hoveredGeoidRef, selectedGeoidRef }) {
  const map = useMap();
  useEffect(() => {
    const onMove = () => {
      // --- tooltip: re-project geographic point to current screen position ---
      const el = tooltipDomRef.current;
      const latLng = lastMouseLatLngRef.current;
      if (el && latLng) {
        const pt = map.latLngToContainerPoint(latLng);
        const mapSize = map.getSize();
        const ttW = el.offsetWidth || 180;
        const ttH = el.offsetHeight || 110;
        const flipX = pt.x + ttW + 20 > mapSize.x;
        const flipY = pt.y + ttH + 20 > mapSize.y;
        el.style.left = `${pt.x + (flipX ? -(ttW + 10) : 14)}px`;
        el.style.top  = `${pt.y + (flipY ? -(ttH + 10) : -10)}px`;
      }
      // --- hover outline: re-apply after canvas redraw ---
      const layer = hoveredLayerRef.current;
      const hGeoid = hoveredGeoidRef.current;
      if (layer && hGeoid) {
        try {
          layer.setStyle(hoverStyle(hGeoid === selectedGeoidRef.current));
          layer.bringToFront();
        } catch (_) {}
      }
    };
    map.on("move", onMove);
    return () => map.off("move", onMove);
  }, [map, tooltipDomRef, lastMouseLatLngRef, hoveredLayerRef, hoveredGeoidRef, selectedGeoidRef]);
  return null;
}

// One recommended-sensor dot. Memoized with a stable center/eventHandlers so a
// parent re-render doesn't call setLatLng/setStyle or re-bind clicks on it.
const QuantumSensorMarker = memo(function QuantumSensorMarker({ sensor, renderer, lang, onSelect }) {
  const isTop3 = (sensor.placement_rank ?? 99) <= 3;
  const center = useMemo(() => [sensor.lat, sensor.lon], [sensor.lat, sensor.lon]);
  const eventHandlers = useMemo(() => ({ click: () => onSelect(sensor.geoid) }), [onSelect, sensor.geoid]);
  return (
    <CircleMarker
      center={center}
      radius={isTop3 ? 8 : 7}
      // renderer must be a top-level option: Leaflet picks a path's renderer
      // when it is added, so a renderer inside pathOptions (applied later via
      // setStyle) is ignored and the dot lands on the pane's default canvas.
      renderer={renderer}
      pathOptions={isTop3 ? QUANTUM_TOP3_STYLE : QUANTUM_SENSOR_STYLE}
      eventHandlers={eventHandlers}
    >
      <Tooltip
        direction="top"
        offset={[0, -10]}
        opacity={0.95}
        className="sensor-tooltip"
      >
        <span style={{ fontWeight: 700 }}>#{sensor.placement_rank}</span>
        {" "}{t(lang, "quantum.recommended_sensor")}
        <br />
        <span style={{ fontSize: 10, opacity: 0.7 }}>
          <span title={t(lang, "quantum.ej_abbr_title")}>{t(lang, "quantum.ej_abbr")}</span>
          : {((sensor.ej_priority || 0) * 100).toFixed(0)}%
          {" | "}
          <span title={t(lang, "quantum.cov_abbr_title")}>{t(lang, "quantum.cov_abbr")}</span>
          : {((sensor.coverage_need || 0) * 100).toFixed(0)}%
        </span>
      </Tooltip>
    </CircleMarker>
  );
});

const GEOJSON_STATUS_KEYS = {
  loading: "map_loading_tracts",
  retrying: "map_waking_server",
  error: "map_tracts_error",
};
// Shapes loaded but predictions still pending: after this long it is most
// likely a cold start waiting on the first compute, so say so.
const PREDICTIONS_SLOW_MS = 6000;

const MapViewContent = forwardRef(
  ({ geojson, geojsonStatus, predictions, onTractSelect, onBackgroundClick, selectedGeoid, searchMarker, sensorMarkers, existingSensors, activeTab = "map" }, _ref) => {
    const { lang } = useContext(LanguageContext);
    const langRef = useRef(lang);
    useEffect(() => { langRef.current = lang; }, [lang]);
    const justClickedRef = useRef(false);
    const onTractSelectRef = useRef(onTractSelect);
    const selectedGeoidRef = useRef(selectedGeoid);
    const predMapRef = useRef({});
    const tooltipDomRef = useRef(null);
    const hoveredLayerRef = useRef(null);
    const hoveredGeoidRef = useRef(null);
    const lastMouseLatLngRef = useRef(null);
    // The choropleth L.GeoJSON, plus a geoid -> polygon index over it so a
    // selection change can restyle two polygons instead of all 6,896.
    const choroplethRef = useRef(null);
    const layerIndexRef = useRef({ owner: null, byGeoid: null });
    // Geoid whose polygon currently carries the selected style.
    const styledSelectionRef = useRef(null);
    // Polygon hover/preview/click is only enabled on the Map tab. We use a ref
    // so the long-lived layer.on() handlers always read the *current* tab
    // (their useCallback has empty deps to keep handler identity stable).
    const activeTabRef = useRef(activeTab);
    const [tooltipData, setTooltipData] = useState(null);
    const [legendExpanded, setLegendExpanded] = useState(false);

    // NOAA HMS smoke polygons still feed the model server-side; the optional
    // map overlay is kept but no longer surfaced in the legend (removed by
    // request), so it stays hidden. setter intentionally omitted.
    const [hmsData, setHmsData] = useState(null);
    const [showHmsSmoke] = useState(false);

    // Fetch the HMS smoke layer only while the overlay is on, refreshing every
    // hour; while it is off nothing would render it, so skip the request.
    useEffect(() => {
      if (!showHmsSmoke) return;
      let alive = true;
      const fetchHms = async () => {
        try {
          const res = await fetch(`${API_BASE}/api/live/hms-smoke`);
          if (!res.ok) return;
          const data = await res.json();
          if (alive) setHmsData(data);
        } catch (e) {
          // Network failures are non-fatal — the layer just won't render.
          console.warn("HMS fetch failed:", e?.message);
        }
      };
      fetchHms();
      const id = setInterval(fetchHms, HMS_REFRESH_MS);
      return () => { alive = false; clearInterval(id); };
    }, [showHmsSmoke]);

    useEffect(() => { onTractSelectRef.current = onTractSelect; }, [onTractSelect]);
    useEffect(() => { activeTabRef.current = activeTab; }, [activeTab]);

    const getTractLayer = useCallback((geoid) => {
      const gj = choroplethRef.current;
      if (!gj || !geoid) return null;
      let idx = layerIndexRef.current;
      if (idx.owner !== gj) {
        const byGeoid = new Map();
        gj.eachLayer((l) => byGeoid.set(normGeoid(l.feature?.properties?.GEOID), l));
        idx = layerIndexRef.current = { owner: gj, byGeoid };
      }
      return idx.byGeoid.get(geoid) ?? null;
    }, []);

    const bringSelectedToFront = useCallback((exceptLayer) => {
      // Neighbours drawn later on the canvas otherwise clip the selected
      // polygon's white border (most visible on the Sensors tab, where hover
      // bringToFront is gated off).
      const layer = getTractLayer(selectedGeoidRef.current);
      if (layer && layer !== exceptLayer) {
        try { layer.bringToFront(); } catch (_) {}
      }
    }, [getTractLayer]);

    // When the user leaves the Map tab, kill any in-flight hover state so it
    // doesn't visually persist (mouseout won't fire because the cursor isn't
    // moving off the polygon — the tab is just being switched).
    useEffect(() => {
      if (activeTab === "map") return;
      setTooltipData(null);
      const layer = hoveredLayerRef.current;
      const geoid = hoveredGeoidRef.current;
      if (layer && geoid) {
        try {
          layer.setStyle(tractStyle(predMapRef.current[geoid], geoid === selectedGeoidRef.current));
        } catch (_) {}
      }
      hoveredLayerRef.current = null;
      hoveredGeoidRef.current = null;
      lastMouseLatLngRef.current = null;
    }, [activeTab]);

    // Dedicated SVG renderer for the quantum-sensor markers so they live in
    // their own high-z-index pane and never get hidden by polygon
    // bringToFront(). SVG (not canvas) is critical here — a canvas in a
    // top-of-stack pane would swallow clicks across the whole map area.
    const sensorSvgRenderer = useMemo(
      () => L.svg({ pane: "quantumSensorsPane" }),
      []
    );

    // Default canvas renderer for the choropleth with a HALF-VIEWPORT padding:
    // polygons are pre-rendered 50% beyond the visible edge, so panning shows
    // painted map instead of blank canvas until the next redraw. tolerance=3
    // widens hover/click hit-testing by 3px (nicer on thin tracts).
    const canvasRenderer = useMemo(
      () => L.canvas({ padding: 0.5, tolerance: 3 }),
      []
    );

    const predMap = useMemo(() => {
      const m = {};
      predictions.forEach((p) => { m[normGeoid(p.geoid)] = p; });
      return m;
    }, [predictions]);

    useEffect(() => { predMapRef.current = predMap; }, [predMap]);

    // Depends on predictions only. react-leaflet's GeoJSON re-applies `style`
    // to every polygon whenever this function's identity changes, so keeping
    // selectedGeoid out of the deps keeps a click from restyling all 6,896
    // tracts; the selection effect below restyles just the two that changed.
    const styleFeature = useCallback((feature) => {
      const geoid = normGeoid(feature.properties?.GEOID);
      return tractStyle(predMap[geoid], geoid === selectedGeoidRef.current);
    }, [predMap]);

    // onEachFeature runs once per feature — uses refs to always access latest state.
    // Tooltip is a custom React overlay (not Leaflet bindTooltip) so it never glitches.
    const onEachFeature = useCallback((feature, layer) => {
      const geoid = normGeoid(feature.properties?.GEOID);
      layer.options.bubblingMouseEvents = false;

      layer.on({
        click: () => {
          // Polygon clicks are only meaningful on the Map tab. On the Sensors
          // tab the polygons are a passive backdrop — only blue dots are
          // interactive. Skipping this also prevents handleDeselect from
          // running (justClickedRef stays false but we don't want any
          // selection-mutating side effects either).
          if (activeTabRef.current !== "map") return;
          justClickedRef.current = true;
          // bubblingMouseEvents=false means no map click follows this one, so
          // the flag would otherwise linger and swallow the next real
          // background click (the first "click away to deselect" did nothing).
          setTimeout(() => { justClickedRef.current = false; }, 0);
          onTractSelectRef.current?.(geoid);
        },

        mouseover: (e) => {
          if (activeTabRef.current !== "map") return;
          hoveredLayerRef.current = e.target;
          hoveredGeoidRef.current = geoid;
          lastMouseLatLngRef.current = e.latlng;
          const pred = predMapRef.current[geoid];
          e.target.setStyle(hoverStyle(geoid === selectedGeoidRef.current));
          e.target.bringToFront();

          if (!pred) return;
          const currentLang = langRef.current;
          const rawName = feature.properties?.NAME ?? geoid;
          // NAME is the English "Census Tract NNNN"; strip it so ES doesn't
          // get "Tracto censal Census Tract NNNN".
          const tractNum = String(rawName).replace(/^Census Tract\s*/i, "");
          const displayName = `${t(currentLang, 'census_tract_prefix')} ${tractNum}`;
          const county = pred.county ? pred.county.replace(/ County$/i, "") : "";
          const pt = e.containerPoint;
          setTooltipData({ name: displayName, pm25: pred.pm25, epaAqi: pred.epa_aqi ?? pm25ToEpaAqi(pred.pm25), color: pred.color, category: pred.category, county, x: pt.x, y: pt.y });
        },

        // Update tooltip position directly on DOM — no React re-render on every mouse move
        mousemove: (e) => {
          if (activeTabRef.current !== "map") return;
          lastMouseLatLngRef.current = e.latlng;
          const el = tooltipDomRef.current;
          if (!el) return;
          const pt = e.containerPoint;
          const mapSize = e.target._map?.getSize();
          const ttW = el.offsetWidth || 180;
          const ttH = el.offsetHeight || 110;
          const flipX = mapSize && pt.x + ttW + 20 > mapSize.x;
          const flipY = mapSize && pt.y + ttH + 20 > mapSize.y;
          el.style.left = `${pt.x + (flipX ? -(ttW + 10) : 14)}px`;
          el.style.top  = `${pt.y + (flipY ? -(ttH + 10) : -10)}px`;
        },

        mouseout: (e) => {
          // mouseout always runs to keep style state clean (it's a no-op if
          // mouseover was suppressed — the polygon is already in default style).
          hoveredLayerRef.current = null;
          hoveredGeoidRef.current = null;
          lastMouseLatLngRef.current = null;
          e.target.setStyle(tractStyle(predMapRef.current[geoid], geoid === selectedGeoidRef.current));
          // The hovered polygon was raised above the selected one; put the
          // selected border back on top.
          bringSelectedToFront(e.target);
          setTooltipData(null);
        },
      });
    }, [bringSelectedToFront]); // stable — all state accessed via refs

    // Only render GeoJSON once BOTH geojson and predictions are ready,
    // so onEachFeature always has a populated predMapRef.
    const hasPolygons =
      geojson?.features?.length > 0 &&
      geojson.features[0]?.geometry &&
      predictions.length > 0;

    // Selection restyle. Runs after the GeoJSON child's own update effect, so
    // when predictions changed in the same commit the all-polygon restyle has
    // already happened and this only corrects the old/new selected tracts and
    // re-applies the hover highlight that restyle wiped. Touches only the
    // choropleth (never HMS smoke or marker layers).
    useEffect(() => {
      selectedGeoidRef.current = selectedGeoid;
      if (!hasPolygons) {
        styledSelectionRef.current = null;
        return;
      }
      const prevGeoid = styledSelectionRef.current;
      const restyle = (geoid) => {
        const layer = getTractLayer(geoid);
        if (!layer) return;
        try { layer.setStyle(tractStyle(predMap[geoid], geoid === selectedGeoid)); } catch (_) {}
      };
      if (prevGeoid && prevGeoid !== selectedGeoid) restyle(prevGeoid);
      if (selectedGeoid) restyle(selectedGeoid);
      styledSelectionRef.current = selectedGeoid || null;
      bringSelectedToFront(null);

      const hovered = hoveredLayerRef.current;
      const hGeoid = hoveredGeoidRef.current;
      if (hovered && hGeoid) {
        try {
          hovered.setStyle(hoverStyle(hGeoid === selectedGeoid));
          hovered.bringToFront();
        } catch (_) {}
      }
    }, [selectedGeoid, predMap, hasPolygons, getTractLayer, bringSelectedToFront]);

    // Basemap fallback: count consecutive tile errors; any successful tile
    // resets the streak. Offline errors are ignored, Esri would fail too.
    const [basemap, setBasemap] = useState("stadia");
    const tileErrorStreakRef = useRef(0);
    const tileEventHandlers = useMemo(() => ({
      tileload: () => { tileErrorStreakRef.current = 0; },
      tileerror: () => {
        if (typeof navigator !== "undefined" && navigator.onLine === false) return;
        tileErrorStreakRef.current += 1;
        // These handlers are only attached to the Stadia layer, so this fires once.
        if (tileErrorStreakRef.current === TILE_ERROR_STREAK_FALLBACK) {
          console.warn("Basemap tiles failing; switching to Esri Dark Gray");
          setBasemap("esri");
        }
      },
    }), []);
    const tiles = BASEMAPS[basemap];

    const handleSensorSelect = useCallback((geoid) => {
      justClickedRef.current = true;
      onTractSelectRef.current?.(geoid);
    }, []);

    const searchCenter = useMemo(
      () => (searchMarker ? [searchMarker.lat, searchMarker.lon] : null),
      [searchMarker?.lat, searchMarker?.lon]
    );

    // existing_sensors entries may lack sensor_id, so key on position too.
    const existingSensorItems = useMemo(
      () => (existingSensors || []).map((s, i) => ({
        key: `existing-${s.sensor_id ?? `${s.lat},${s.lon}`}-${i}`,
        center: [s.lat, s.lon],
      })),
      [existingSensors]
    );

    // Flag yesterday's proxy file, and anything not dated today even if the
    // backend did not flag it.
    const hmsIsStale = Boolean(hmsData?.data_date) &&
      (hmsData.is_proxy === true || hmsData.data_date !== todayUtcIso());

    // The shapes alone can't be drawn (hasPolygons needs predictions too), so
    // keep an overlay up until predictions arrive instead of a bare basemap.
    const predictionsPending = geojsonStatus === "ready" && predictions.length === 0;
    const [predictionsSlow, setPredictionsSlow] = useState(false);
    useEffect(() => {
      if (!predictionsPending) { setPredictionsSlow(false); return; }
      const id = setTimeout(() => setPredictionsSlow(true), PREDICTIONS_SLOW_MS);
      return () => clearTimeout(id);
    }, [predictionsPending]);
    const statusKey = predictionsPending
      ? (predictionsSlow ? "map_waking_server" : "map_loading_tracts")
      : GEOJSON_STATUS_KEYS[geojsonStatus];

    return (
      <div style={{ position: "relative", height: "100%", width: "100%" }}>
        <MapContainer
          center={TEXAS_CENTER}
          zoom={TEXAS_ZOOM}
          style={{ height: "100%", width: "100%" }}
          zoomControl={true}
          zoomSnap={0}
          zoomDelta={1}
          scrollWheelZoom={false}
          preferCanvas={true}
          renderer={canvasRenderer}
          touchZoom={true}
          tap={true}
          zoomAnimation={true}
          zoomAnimationThreshold={8}
          fadeAnimation={true}
          inertia={true}
          inertiaDeceleration={3000}
          inertiaMaxSpeed={1500}
          maxZoom={13}
          minZoom={4}
          maxBounds={[[23.5, -111.5], [39.5, -87.5]]}
          maxBoundsViscosity={0.6}
        >
          <TileLayer
            key={basemap}
            url={tiles.url}
            attribution={tiles.attribution}
            zIndex={1}
            keepBuffer={8}
            updateWhenZooming={true}
            updateWhenIdle={false}
            updateInterval={150}
            crossOrigin="anonymous"
            noWrap={true}
            maxNativeZoom={tiles.maxNativeZoom}
            eventHandlers={basemap === "stadia" ? tileEventHandlers : undefined}
          />

          <FlyToHandler target={searchMarker} />
          <AdaptiveWheelZoom />
          <MapLifecycle />
          <BackgroundClickHandler onBackgroundClick={onBackgroundClick} justClickedRef={justClickedRef} />
          <MapMoveHandler
            tooltipDomRef={tooltipDomRef}
            lastMouseLatLngRef={lastMouseLatLngRef}
            hoveredLayerRef={hoveredLayerRef}
            hoveredGeoidRef={hoveredGeoidRef}
            selectedGeoidRef={selectedGeoidRef}
          />

          {hasPolygons && (
            <GeoJSON
              ref={choroplethRef}
              data={geojson}
              style={styleFeature}
              onEachFeature={onEachFeature}
            />
          )}

          {/* NOAA HMS smoke polygons — own pane between choropleth (z=400) and
              sensor markers (z=600). Pointer-events:none so the smoke overlay
              never steals clicks from the choropleth underneath. */}
          {showHmsSmoke && hmsData?.features?.length > 0 && (
            <Pane name="hmsSmokePane" style={{ zIndex: 450, pointerEvents: "none" }}>
              <GeoJSON
                key={`hms-${hmsData.data_date}-${hmsData.count}`}
                data={hmsData}
                style={hmsStyle}
              />
            </Pane>
          )}

          {searchCenter && (
            <Circle
              center={searchCenter}
              radius={120}
              interactive={false}
              bubblingMouseEvents={false}
              className="search-marker-circle"
              pathOptions={SEARCH_MARKER_STYLE}
            />
          )}

          {/* Existing PurpleAir sensors (grey dots) */}
          {existingSensorItems.map((s) => (
            <CircleMarker
              key={s.key}
              center={s.center}
              radius={4}
              pathOptions={EXISTING_SENSOR_STYLE}
              interactive={false}
              bubblingMouseEvents={false}
            />
          ))}

          {/* Quantum sensor placement markers — rendered as SVG (not canvas)
              in their OWN pane with z-index above overlayPane so polygon
              bringToFront() calls can never push them behind. The pane DIV
              itself has pointer-events:none so empty space passes clicks
              through; only the SVG paths (with their default
              pointer-events:visiblePainted) capture clicks on the actual dots. */}
          <Pane name="quantumSensorsPane" style={{ zIndex: 600, pointerEvents: "none" }}>
            {sensorMarkers && sensorMarkers.map((s) => (
              <QuantumSensorMarker
                key={`sensor-${s.geoid}`}
                sensor={s}
                renderer={sensorSvgRenderer}
                lang={lang}
                onSelect={handleSensorSelect}
              />
            ))}
          </Pane>


          <button
            ref={isolateFromMap}
            type="button"
            className="mobile-legend-toggle"
            aria-label={t(lang, 'legend_toggle_aria')}
            aria-expanded={legendExpanded}
            onClick={() => setLegendExpanded(v => !v)}
          >
            {legendExpanded ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="9" />
                <line x1="12" y1="11" x2="12" y2="17" />
                <circle cx="12" cy="7.5" r="0.9" fill="currentColor" stroke="none" />
              </svg>
            )}
          </button>
          <div ref={isolateFromMap} className={`map-legend${legendExpanded ? ' expanded' : ''}`}>
            {/* The title is uppercased; keep the unit as-is or µg reads as "MG". */}
            <div className="legend-title">PM2.5 <span style={{ textTransform: "none" }}>µg/m³</span></div>
            {BREAKPOINTS.map((b) => (
              <div className="legend-row" key={b.category}>
                <div className="legend-swatch" style={{ background: b.color }} />
                <span className="legend-category">{translateCategory(lang, b.category)}</span>
                <span className="legend-range">{b.label}</span>
              </div>
            ))}
            {/* Smoke is dated so an older analysis is never read as today's. */}
            {showHmsSmoke && hmsData?.data_date && (
              <div className="legend-hms-date">
                {lang === "es" ? "Humo NOAA HMS: " : "NOAA HMS smoke: "}{hmsData.data_date}
                {hmsIsStale && (lang === "es" ? " (el más reciente, no es de hoy)" : " (latest available, not today's)")}
              </div>
            )}
          </div>
        </MapContainer>

        {/* Always mounted so screen readers pick up the status change. */}
        <div className="map-status-overlay" role="status" aria-live="polite">
          {statusKey && (
            <div className={`map-status-card${geojsonStatus === "error" ? " is-error" : ""}`}>
              <span className="map-status-spinner" aria-hidden="true" />
              <span>{t(lang, statusKey)}</span>
            </div>
          )}
        </div>

        {/* Custom tooltip — positioned absolutely over the map, never glitches */}
        {tooltipData && (
          <div
            ref={tooltipDomRef}
            className="custom-tract-tooltip"
            style={{ left: tooltipData.x + 14, top: tooltipData.y - 10 }}
          >
            <div className="ctt-name">{tooltipData.name}</div>
            <div className="ctt-pm25" style={{ color: tooltipData.color }}>
              {tooltipData.pm25} <span className="ctt-unit">µg/m³</span>
              <span className="ctt-aqi"> · {t(lang, 'tooltip.aqi_label')} {tooltipData.epaAqi}</span>
            </div>
            <div className="ctt-cat">{translateCategory(lang, tooltipData.category)}</div>
            <div className="ctt-modeled">{t(lang, 'tooltip.modeled_note')}</div>
            {tooltipData.county && (
              <div className="ctt-county">{tooltipData.county} {t(lang, 'tooltip.county_suffix')}</div>
            )}
          </div>
        )}
      </div>
    );
  }
);

MapViewContent.displayName = "MapViewContent";

// memo: App re-renders (search input, weather, panel state) leave the map's
// props unchanged, so they no longer re-render the whole Leaflet tree.
export default memo(forwardRef((props, ref) => <MapViewContent {...props} ref={ref} />));
