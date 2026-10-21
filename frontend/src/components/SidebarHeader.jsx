import { useContext } from "react";
import { LanguageContext } from "../App";
import { t } from "../i18n";

/**
 * Shared sidebar header used by both the Map tab (SidePanel) and the Sensors
 * tab (SensorPlacement). Renders the brand row, the visit counter, the tagline
 * (which morphs into the selected-tract identifier), and the EN | ES language
 * toggle. When a tract is selected, a back button appears to deselect.
 */
export default function SidebarHeader({ selectedTract, visitCount, onDeselect, statewide = true }) {
  const { lang, setLang } = useContext(LanguageContext);
  const displayName = statewide
    ? (lang === "es" ? "Todo Texas" : "All of Texas")
    : (lang === "es" ? "Resumen de la región" : "Region Overview");
  const countyName = selectedTract?.county
    ? selectedTract.county.replace(/ County$/i, "")
    : "";

  function toggleLang(newLang) {
    setLang(newLang);
  }

  return (
    <div className="sidebar-header">
      {selectedTract && onDeselect && (
        <button className="back-btn" onClick={onDeselect} aria-label={t(lang, "back_to", displayName)}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <line x1="19" y1="12" x2="5" y2="12" />
            <polyline points="12 19 5 12 12 5" />
          </svg>
          <span>{t(lang, "back_to", displayName)}</span>
        </button>
      )}
      <div className="brand">
        <div className="brand-left">
          <div className="brand-icon" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2v4" />
              <path d="M12 18v4" />
              <path d="M4.93 4.93l2.83 2.83" />
              <path d="M16.24 16.24l2.83 2.83" />
              <path d="M2 12h4" />
              <path d="M18 12h4" />
              <path d="M4.93 19.07l2.83-2.83" />
              <path d="M16.24 7.76l2.83-2.83" />
              <circle cx="12" cy="12" r="3.5" fill="currentColor" stroke="none" />
            </svg>
          </div>
          <span className="brand-name">Shared Skies Initiative</span>
        </div>
        {visitCount != null && (
          <div className="visit-counter-header" title={t(lang, "visits_tooltip")}>
            <span data-num>{t(lang, "visits_label", visitCount.toLocaleString())}</span>
          </div>
        )}
      </div>
      <div className="brand-tagline">
        {selectedTract
          ? `${lang === "es" ? "Tracto" : "Tract"} ${selectedTract.geoid?.slice(-6)}${countyName ? ` · ${countyName}` : ""}`
          : t(lang, "brand_tagline_all")}
      </div>

      <div className="lang-toggle-wrap">
        <div className="lang-toggle" role="group" aria-label={t(lang, "lang_selector_aria")}>
          <button
            className={`lang-btn ${lang === "en" ? "active" : ""}`}
            onClick={() => toggleLang("en")}
            aria-pressed={lang === "en"}
          >EN</button>
          <button
            className={`lang-btn ${lang === "es" ? "active" : ""}`}
            onClick={() => toggleLang("es")}
            aria-pressed={lang === "es"}
          >ES</button>
        </div>
      </div>
    </div>
  );
}
