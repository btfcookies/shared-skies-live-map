import { BREAKPOINTS, pm25ToGaugePct, pm25ToEpaAqi } from "../utils/aqi.js";
import { useContext, useEffect, useMemo, useState, memo } from "react";
import { LanguageContext } from "../App";
import { t, translateCategory, translateHealth, formatPercentile } from "../i18n";
import SidebarHeader from "./SidebarHeader.jsx";

function fmt(val, decimals = 1) {
  if (val == null || isNaN(val)) return "—";
  return Number(val).toFixed(decimals);
}

const DEFAULT_REFRESH_MIN = 15;

// Re-renders itself once a minute so the age stays honest between polls,
// without re-rendering the rest of the panel.
function UpdatedAgo({ generatedAt, lang }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 60 * 1000);
    return () => clearInterval(id);
  }, []);
  const ts = Date.parse(generatedAt);
  if (Number.isNaN(ts)) return null;
  // Clamp: a client clock running behind the server would otherwise go negative.
  const min = Math.max(0, Math.floor((now - ts) / 60000));
  return min < 1 ? t(lang, "updated_just_now") : t(lang, "updated_minutes_ago", min);
}

function Credits({ lang }) {
  const credits = t(lang, "credits");
  return (
    <div className="footer-row footer-data">
      {t(lang, "credits_label")}:{" "}
      {credits.map((segments, i) => (
        <span className="credit-item" key={i}>
          {i > 0 && " · "}
          {segments.map((seg, j) => (typeof seg === "string" ? seg : (
            <a key={j} href={seg.href} target="_blank" rel="noreferrer">{seg.text}</a>
          )))}
        </span>
      ))}
    </div>
  );
}

const PM25Hero = memo(function PM25Hero({ pm25, epaAqi, color, category, lang }) {
  const gaugePct = pm25ToGaugePct(pm25);
  const aqi = epaAqi ?? pm25ToEpaAqi(pm25);
  return (
    <>
      <div className="pm25-hero">
        <div className="pm25-label">{t(lang, "current_pm25")}</div>
        <div className="pm25-number-row">
          <div
            className="pm25-number"
            style={{ color, textShadow: `0 0 40px ${color}55` }}
            data-num
          >
            {fmt(pm25, 2)}
          </div>
          <div className="pm25-unit">µg/m³</div>
        </div>
        <div className="pm25-aqi-equiv" data-num>{t(lang, "aqi_equiv", aqi)}</div>
        <div
          className="aqi-badge"
          style={{
            background: `${color}1f`,
            color,
            borderColor: `${color}55`,
          }}
        >
          {translateCategory(lang, category)}
        </div>
      </div>
      <div className="gauge-wrap">
        <div className="gauge-bar-bg">
          <div
            className="gauge-bar-fill"
            style={{ width: `${gaugePct}%`, background: color, color }}
          />
        </div>
        <div className="gauge-labels">
          <span>{translateCategory(lang, "Good")}</span>
          <span>{translateCategory(lang, "Moderate")}</span>
          <span>{translateCategory(lang, "Elevated")}</span>
          <span>{translateCategory(lang, "High")}</span>
        </div>
      </div>
    </>
  );
});

const WeatherWidget = memo(function WeatherWidget({ weather, isLoading, lang }) {
  return (
    <>
      <div className="section-header">
        <span>{t(lang, "current_conditions")}</span>
        {isLoading && (
          <span className="weather-loading-badge">{t(lang, "tooltip.updating")}</span>
        )}
      </div>
      <div className="weather-grid">
        <div className="weather-item">
          <div className="weather-item-label">{t(lang, "temperature")}</div>
          <div className="weather-item-value" data-num>
            {!weather ? "—" : `${fmt(weather.temperature, 0)}°`}
          </div>
          <div className="weather-item-sub">Fahrenheit</div>
        </div>
        <div className="weather-item">
          <div className="weather-item-label">{t(lang, "humidity")}</div>
          <div className="weather-item-value" data-num>
            {!weather ? "—" : `${fmt(weather.humidity, 0)}%`}
          </div>
          <div className="weather-item-sub">{lang === "es" ? "Relativa" : "Relative"}</div>
        </div>
        <div className="weather-item">
          <div className="weather-item-label">{t(lang, "pressure")}</div>
          <div className="weather-item-value" data-num>
            {!weather ? "—" : fmt(weather.pressure, 0)}
          </div>
          <div className="weather-item-sub">hPa</div>
        </div>
        <div className="weather-item">
          <div className="weather-item-label">{t(lang, "wind")}</div>
          <div className="weather-item-value" data-num>
            {!weather ? "—" : fmt(weather.wind_speed, 0)}
          </div>
          <div className="weather-item-sub">mph</div>
        </div>
      </div>
    </>
  );
});

const EJ_FIELD_KEYS = [
  "ejf_score",
  "pct_people_of_color",
  "pct_low_income",
  "traffic_proximity",
  "diesel_pm_proximity",
  "superfund_proximity",
];

function EJContext({ tract, lang }) {
  const ejFields = useMemo(() => {
    const labels = t(lang, "ej_fields");
    return EJ_FIELD_KEYS.map((key) => ({ key, ...labels[key] }));
  }, [lang]);

  const hasAny = ejFields.some((f) => tract[f.key] != null);
  if (!hasAny) return null;

  return (
    <>
      <div className="section-header">{t(lang, "ej_context")}</div>
      <div className="ej-list">
        {ejFields.map(({ key, label, desc }) => {
          const val = tract[key];
          if (val == null) return null;
          const pct = Math.min(100, Math.max(0, val));
          return (
            <div className="ej-item" key={key}>
              <div className="ej-item-header">
                <span className="ej-item-name">{label}</span>
                <span className="ej-item-value" data-num>
                  {formatPercentile(lang, val)}
                </span>
              </div>
              <div className="ej-bar-bg">
                <div className="ej-bar-fill" style={{ width: `${pct}%` }} />
              </div>
              <div className="ej-context">{desc}</div>
            </div>
          );
        })}
      </div>
    </>
  );
}

function DistributionSummary({ tracts, lang }) {
  const stats = useMemo(() => {
    if (!tracts?.length) return null;
    const counts = {};
    BREAKPOINTS.forEach((b) => { counts[b.category] = 0; });
    tracts.forEach((tt) => {
      if (tt.category) counts[tt.category] = (counts[tt.category] ?? 0) + 1;
    });
    return { counts, total: tracts.length };
  }, [tracts]);

  if (!stats) return null;
  const { counts, total } = stats;

  return (
    <>
      <div className="section-header">{t(lang, "tract_distribution")}</div>
      <div className="distribution">
        {BREAKPOINTS.map((b) => {
          const n = counts[b.category] ?? 0;
          if (n === 0) return null;
          return (
            <div className="dist-row" key={b.category}>
              <div className="dist-dot" style={{ background: b.color, color: b.color }} />
              <span className="dist-label">{translateCategory(lang, b.category)}</span>
              <span className="dist-count" data-num>{n.toLocaleString()}</span>
              <span className="dist-pct" data-num>{Math.round((n / total) * 100)}%</span>
            </div>
          );
        })}
      </div>
    </>
  );
}

function LoadingSkeleton({ lang }) {
  return (
    <div className="loading-wrap" aria-busy="true" aria-live="polite">
      <div className="sk-row">
        <div className="skeleton sk-h" style={{ width: "40%" }} />
        <div className="skeleton sk-num" />
      </div>
      <div className="skeleton sk-bar" style={{ marginTop: 6 }} />
      <div className="sk-grid" style={{ marginTop: 12 }}>
        <div className="skeleton sk-card" />
        <div className="skeleton sk-card" />
        <div className="skeleton sk-card" />
        <div className="skeleton sk-card" />
      </div>
      <div className="loading-text">{t(lang, "loading_predictions")}</div>
    </div>
  );
}

export default function SidePanel({
  predictions,
  selectedTract,
  localWeather,
  onDeselect,
  refreshing = false,
  weatherLoading = false,
  error,
  statewide = false,
  visitCount,
}) {
  const { lang } = useContext(LanguageContext);
  const sources = predictions?.data_sources;
  const refreshMin = sources?.refresh_interval_min ?? DEFAULT_REFRESH_MIN;

  return (
    <aside className="sidebar-content">
      <SidebarHeader
        selectedTract={selectedTract}
        visitCount={visitCount}
        onDeselect={onDeselect}
        statewide={statewide}
      />

      {/* Once data exists it stays on screen; a failed background refresh just
          leaves the "Updated N min ago" line to age. */}
      {error && !predictions && (
        <div className="error-wrap" role="status">
          <strong>{t(lang, "predictions_error_title")}</strong>
          <br />
          {t(lang, "predictions_error_body")}
        </div>
      )}

      {!predictions && <LoadingSkeleton lang={lang} />}

      {selectedTract && (
        <>
          <PM25Hero
            pm25={selectedTract.pm25}
            epaAqi={selectedTract.epa_aqi}
            color={selectedTract.color}
            category={selectedTract.category}
            lang={lang}
          />
          <div className="health-card">
            {translateHealth(lang, selectedTract.category) || selectedTract.health_msg}
          </div>
          <WeatherWidget
            weather={localWeather || predictions?.weather}
            isLoading={weatherLoading}
            lang={lang}
          />
          {localWeather && (
            <div className="weather-source-note">
              {lang === "es"
                ? "Condiciones actuales cerca de este tracto · se actualizan cada hora"
                : "Current conditions near this tract · updates hourly"}
            </div>
          )}
          <EJContext tract={selectedTract} lang={lang} />
        </>
      )}

      {!selectedTract && predictions && (
        <div className="overview-body">
          <PM25Hero
            pm25={predictions.avg_pm25}
            epaAqi={predictions.avg_epa_aqi}
            color={predictions.avg_info?.color ?? "#10b981"}
            category={predictions.avg_info?.category ?? "Good"}
            lang={lang}
          />
          <div className="health-card">
            {translateHealth(lang, predictions.avg_info?.category) || predictions.avg_info?.health_msg}
          </div>
          <div className="prompt-hint">{t(lang, "use_search")}</div>
          <WeatherWidget weather={predictions.weather} isLoading={false} lang={lang} />
          <DistributionSummary tracts={predictions.tracts} lang={lang} />
        </div>
      )}

      <div className="sidebar-footer">
        {sources && (
          <div
            className={`footer-row live-badge ${sources.using_live_neighbors ? "is-live" : "is-fallback"}`}
            role="status"
          >
            {sources.using_live_neighbors
              ? <>
                  {t(lang, "live_badge_live")}
                  {sources.live_purpleair_sensors > 0 && (
                    <> · {t(lang, "live_sensors_note", sources.live_purpleair_sensors.toLocaleString())}</>
                  )}
                </>
              : t(lang, "live_badge_fallback")}
          </div>
        )}
        {predictions?.generated_at && (
          <div className="footer-row">
            <span className="last-updated">
              <UpdatedAgo generatedAt={predictions.generated_at} lang={lang} />
              {" · "}{t(lang, "refresh_note", refreshMin)}
            </span>
            {refreshing && (
              <span className="refresh-indicator" aria-live="polite">{t(lang, "refreshing")}</span>
            )}
          </div>
        )}
        <Credits lang={lang} />
        <div className="footer-row footer-built">
          {lang === "es" ? "Creado por" : "Built by"}{" "}
          <a
            href="https://sharedskiesinitiative.org/real-time-map"
            target="_blank"
            rel="noreferrer"
          >
            Shared Skies Initiative
          </a>{" "}
          · {lang === "es" ? "Cobertura en todo Texas" : "Texas-wide Coverage"}
        </div>
      </div>
    </aside>
  );
}
