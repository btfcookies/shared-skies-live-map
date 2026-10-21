import { useContext, useEffect, useState, memo } from "react";
import { LanguageContext } from "../App";
import { t } from "../i18n";
import SidebarHeader from "./SidebarHeader.jsx";

// Only used if an older backend omits methods.quantum_annealing.weights;
// mirrors W_COVERAGE/W_ERROR/W_EJ/W_PM25 in backend/quantum/qubo_solver.py.
const DEFAULT_WEIGHTS = { coverage: 0.30, error: 0.20, ej: 0.30, pm25: 0.20 };
const WEIGHT_KEYS = ["coverage", "error", "ej", "pm25"];
const DEFAULT_TOTAL_TRACTS = 6896;
const DEFAULT_NUM_SENSORS = 25;
const SLOW_NOTICE_MS = 8000;

const SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹";

// "5.7 × 10⁷⁰" for n-choose-k, summed in log space so it never overflows.
function formatCombinations(n, k) {
  if (!(n >= k && k > 0)) return null;
  let lg = 0;
  for (let i = 1; i <= k; i++) lg += Math.log10((n - k + i) / i);
  let exp = Math.floor(lg);
  let mantissa = Math.pow(10, lg - exp);
  if (mantissa.toFixed(1) === "10.0") { mantissa = 1; exp += 1; }
  const sup = String(exp).split("").map((d) => SUPERSCRIPT_DIGITS[d]).join("");
  return `${mantissa.toFixed(1)} × 10${sup}`;
}

// Weights as whole-number percentages of their sum, so the copy and the bars
// stay right even if the solver's weights don't add up to exactly 1.
function weightPercents(weights) {
  const w = weights && WEIGHT_KEYS.every((k) => typeof weights[k] === "number")
    ? weights
    : DEFAULT_WEIGHTS;
  const total = WEIGHT_KEYS.reduce((sum, k) => sum + w[k], 0) || 1;
  return Object.fromEntries(WEIGHT_KEYS.map((k) => [k, Math.round((w[k] / total) * 100)]));
}

const MetricCard = memo(function MetricCard({ label, value, sub, accent }) {
  return (
    <div className="qm-metric-card" style={accent ? { color: accent } : {}}>
      <div className="qm-metric-label">{label}</div>
      <div className="qm-metric-value" style={accent ? { color: accent } : {}} data-num>
        {value}
      </div>
      {sub && <div className="qm-metric-sub">{sub}</div>}
    </div>
  );
});

function EJQuartileBar({ label, data }) {
  if (!data) return null;
  return (
    <div className="qm-ej-bar-row">
      <span className="qm-ej-bar-label">{label}</span>
      <div className="qm-ej-bar-track">
        <div className="qm-ej-bar-fill" style={{ width: `${data.pct}%` }} />
      </div>
      <span className="qm-ej-bar-pct" data-num>{data.pct}%</span>
      {data.total != null && (
        <span className="qm-ej-bar-detail" data-num>
          {data.covered}/{data.total}
        </span>
      )}
    </div>
  );
}

export default function SensorPlacement({
  quantumData,
  loading,
  error,
  onRetry,
  onViewSensor,
  selectedTract,
  onDeselect,
  visitCount,
}) {
  const { lang } = useContext(LanguageContext);
  // The first solve after a cold start is slow; say so instead of spinning
  // silently.
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    if (!loading) { setSlow(false); return; }
    const id = setTimeout(() => setSlow(true), SLOW_NOTICE_MS);
    return () => clearTimeout(id);
  }, [loading]);

  const sharedHeader = (
    <SidebarHeader
      selectedTract={selectedTract}
      visitCount={visitCount}
      onDeselect={onDeselect}
      statewide={true}
    />
  );

  if (loading) {
    return (
      <>
        {sharedHeader}
        <div className="qm-loading" aria-busy="true" aria-live="polite">
          <div className="qm-loading-icon">
            <div className="qm-atom">
              <div className="qm-orbit qm-orbit-1" />
              <div className="qm-orbit qm-orbit-2" />
              <div className="qm-orbit qm-orbit-3" />
              <div className="qm-nucleus" />
            </div>
          </div>
          <div className="qm-loading-text">{t(lang, "quantum.loading")}</div>
          <div className="qm-loading-detail">{t(lang, "quantum.loading_detail")}</div>
          {slow && <div className="qm-loading-detail">{t(lang, "quantum.loading_slow")}</div>}
        </div>
      </>
    );
  }

  if (error) {
    return (
      <>
        {sharedHeader}
        <div className="qm-error">
          <strong>{t(lang, "quantum.error")}</strong>
          <code>{error}</code>
          {onRetry && (
            <button type="button" className="qm-retry-btn" onClick={onRetry}>
              {t(lang, "quantum.retry")}
            </button>
          )}
        </div>
      </>
    );
  }

  const quantum = quantumData?.methods?.quantum_annealing;
  if (!quantum) return sharedHeader;

  const quantumCov = quantum.coverage || {};
  const quantumTracts = quantum.selected_tracts || [];
  const ejQuartiles = quantumCov.coverage_by_ej_quartile || {};
  // Top-level num_sensors is the requested k; the method's num_sensors is how
  // many well-spaced sites the solver could actually place.
  const numSensors = quantumData.num_sensors ?? DEFAULT_NUM_SENSORS;
  const numPlaced = quantum.num_sensors ?? quantumTracts.length;
  const totalTracts = quantumData.num_total_tracts ?? DEFAULT_TOTAL_TRACTS;
  const numCandidates = quantum.num_candidates;
  const weights = weightPercents(quantum.weights);
  // The solver's method_display is English-only, so other languages get the
  // localized label.
  const methodDisplay = (lang === "en" && quantum.method_display) || t(lang, "quantum.quantum_annealing");

  return (
    <div className="qm-container">
      {sharedHeader}

      <div className="qm-header">
        <div className="qm-title">{t(lang, "quantum.title")}</div>
        <div className="qm-subtitle">{t(lang, "quantum.subtitle")}</div>
      </div>

      {quantumData.num_existing_sensors > 0 && (
        <div className="qm-existing-info">
          <span className="qm-existing-dot" />
          <span data-num>
            {t(lang, "quantum.existing_sensors", quantumData.num_existing_sensors.toLocaleString())}
          </span>
          <span className="qm-existing-hint">{t(lang, "quantum.existing_hint")}</span>
        </div>
      )}

      <div className="qm-metrics-grid">
        <MetricCard
          label={t(lang, "quantum.num_sensors")}
          value={numPlaced}
          sub={t(lang, "quantum.new_placements")}
          accent="#38bdf8"
        />
        <MetricCard
          label={t(lang, "quantum.coverage")}
          value={`${quantumCov.pct_covered}%`}
          sub={
            quantumCov.new_covered != null
              ? `+${quantumCov.new_covered} ${t(lang, "quantum.new_suffix")}`
              : `${quantumCov.covered_count}/${quantumCov.total_tracts}`
          }
          accent="#10b981"
        />
        <MetricCard
          label={t(lang, "quantum.avg_distance")}
          value={`${quantumCov.avg_distance_miles} mi`}
          accent="#fbbf24"
        />
        <MetricCard
          label={t(lang, "quantum.max_gap")}
          value={`${quantumCov.max_distance_miles} mi`}
          accent="#ef4444"
        />
      </div>

      <div className="section-header">{t(lang, "quantum.method")}</div>
      <div className="qm-method">
        <div className="qm-method-name">{methodDisplay}</div>
        {numCandidates > 0 && (
          <div className="qm-method-note" data-num>{t(lang, "quantum.candidate_pool", numCandidates)}</div>
        )}
        {typeof quantum.feasible === "boolean" && (
          <div className={`qm-method-note ${quantum.feasible ? "is-ok" : "is-warn"}`}>
            {quantum.feasible
              ? t(lang, "quantum.feasible_yes", numSensors)
              : t(lang, "quantum.feasible_no", numPlaced, numSensors)}
          </div>
        )}
        <div className="qm-method-sub">{t(lang, "quantum.weights")}</div>
        <div className="qm-ej-bars qm-weights">
          {WEIGHT_KEYS.map((k) => (
            <EJQuartileBar
              key={k}
              label={t(lang, `quantum.weight_names.${k}`)}
              data={{ pct: weights[k] }}
            />
          ))}
        </div>
      </div>

      <div className="section-header">{t(lang, "quantum.coverage_by_ej")}</div>
      <div className="qm-ej-bars">
        <EJQuartileBar label={t(lang, "quantum.q1_low")}  data={ejQuartiles["Q1 (Low EJ)"]} />
        <EJQuartileBar label={t(lang, "quantum.q2")}      data={ejQuartiles["Q2"]} />
        <EJQuartileBar label={t(lang, "quantum.q3")}      data={ejQuartiles["Q3"]} />
        <EJQuartileBar label={t(lang, "quantum.q4_high")} data={ejQuartiles["Q4 (High EJ)"]} />
      </div>

      <div className="section-header">{t(lang, "quantum.recommended_locations")}</div>
      <div className="qm-sensor-list">
        {quantumTracts.slice(0, numSensors).map((s) => {
          const county = s.county ? s.county.replace(/ County$/i, "") : "";
          return (
            <div
              className="qm-sensor-item"
              key={s.geoid}
              onClick={() => onViewSensor?.(s)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && onViewSensor?.(s)}
              aria-label={t(lang, "quantum.view_tract", s.geoid?.slice(-6), county)}
            >
              <div className="qm-sensor-rank" data-num>#{s.placement_rank}</div>
              <div className="qm-sensor-info">
                <div className="qm-sensor-geoid" data-num>
                  {t(lang, "census_tract_prefix")} {s.geoid?.slice(-6)}
                </div>
                <div className="qm-sensor-county">{county} {t(lang, "tooltip.county_suffix")}</div>
              </div>
              <div className="qm-sensor-scores">
                <div className="qm-sensor-score-row">
                  <span className="qm-score-label" title={t(lang, "quantum.ej_abbr_title")}>
                    {t(lang, "quantum.ej_abbr")}
                  </span>
                  <div className="qm-score-bar-bg">
                    <div
                      className="qm-score-bar-fill ej"
                      style={{ width: `${Math.min(100, (s.ej_priority || 0) * 100)}%` }}
                    />
                  </div>
                </div>
                <div className="qm-sensor-score-row">
                  <span className="qm-score-label" title={t(lang, "quantum.cov_abbr_title")}>
                    {t(lang, "quantum.cov_abbr")}
                  </span>
                  <div className="qm-score-bar-bg">
                    <div
                      className="qm-score-bar-fill cov"
                      style={{ width: `${Math.min(100, (s.coverage_need || 0) * 100)}%` }}
                    />
                  </div>
                </div>
              </div>
              <div className="qm-sensor-composite" data-num>
                {((s.composite_score || 0) * 100).toFixed(0)}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ padding: "0 22px", display: "flex", flexDirection: "column", gap: 12 }}>
        <div className="guide-explainer">
          <h4>{t(lang, "quantum.tab_intro_title")}</h4>
          {t(lang, "quantum.tab_intro_body", {
            k: numSensors,
            n: totalTracts.toLocaleString("en-US"),
            existing: quantumData.num_existing_sensors > 0
              ? quantumData.num_existing_sensors.toLocaleString("en-US")
              : null,
          })}
        </div>

        <div className="guide-explainer">
          <h4>{t(lang, "quantum.what_is")}</h4>
          {t(lang, "quantum.what_is_body", {
            k: numSensors,
            n: totalTracts.toLocaleString("en-US"),
            total: formatCombinations(totalTracts, numSensors),
            candidates: numCandidates > numSensors ? numCandidates.toLocaleString("en-US") : null,
            candidateTotal: formatCombinations(numCandidates, numSensors),
          })}
        </div>

        <div className="guide-explainer">
          <h4>{t(lang, "quantum.why_ej")}</h4>
          {t(lang, "quantum.why_ej_body", weights)}
        </div>
      </div>
    </div>
  );
}
