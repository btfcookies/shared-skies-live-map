import React, { useState, useRef, useEffect, useContext } from "react";
import "./SearchBar.css";
import { LanguageContext } from "../App";
import { t } from "../i18n";

const API_BASE = import.meta.env.VITE_API_URL || "";
// Matches the backend contract: q is capped at 100 chars (422 beyond that).
const MAX_QUERY_LEN = 100;
const MIN_SUGGEST_LEN = 3;
const SUGGEST_DEBOUNCE_MS = 350;

// mode=suggest is Photon only; mode=submit may also fall back to Nominatim,
// which has a strict usage policy, so it is reserved for explicit searches.
// Returns [] for any non-2xx (429 rate limit, 422, 5xx) so callers can treat
// it as "no results" without special cases.
async function fetchGeocode(query, mode, lang, signal) {
  const res = await fetch(
    `${API_BASE}/api/geocode?q=${encodeURIComponent(query)}&mode=${mode}`,
    { headers: { "Accept-Language": lang === "es" ? "es" : "en" }, signal }
  );
  if (!res.ok) {
    const err = new Error(`Geocode error ${res.status}`);
    err.status = res.status;
    throw err;
  }
  const results = await res.json();
  return Array.isArray(results) ? results : [];
}

const PinIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
    <circle cx="12" cy="10" r="3" />
  </svg>
);

const CompassIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="10" />
    <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
  </svg>
);

const SearchIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="search-input-icon">
    <circle cx="11" cy="11" r="7" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const AlertIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </svg>
);

export default function SearchBar({ onSearch, predictionsReady = true }) {
  const { lang } = useContext(LanguageContext);
  const [searchInput, setSearchInput] = useState("");
  const [searchType, setSearchType] = useState("address");
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [suggestLoading, setSuggestLoading] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  // Set when suggestions are unavailable (rate limit / server error), so the
  // dropdown doesn't claim "address not found".
  const [suggestUnavailable, setSuggestUnavailable] = useState(false);
  // The query the current suggestions belong to. Enter only reuses them, and
  // the dropdown only says "not found", when they match what is typed.
  const [suggestionsFor, setSuggestionsFor] = useState("");
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const searchRef = useRef(null);
  const inputRef = useRef(null);
  const timeoutRef = useRef(null);
  // Picking a suggestion writes its name into the input; that write must not
  // start a new lookup and reopen the dropdown.
  const skipNextQueryRef = useRef(false);

  const abortControllerRef = useRef(null);

  // Debounced typo-tolerant suggestions (backend proxies Photon w/ Texas bias;
  // no need to append ", Texas" — the server scopes results to the TX bbox).
  useEffect(() => {
    const query = searchInput.trim();
    if (skipNextQueryRef.current) {
      skipNextQueryRef.current = false;
      setSuggestLoading(false);
      return;
    }
    if (searchType !== "address" || query.length < MIN_SUGGEST_LEN) {
      setSuggestions([]);
      setSuggestionsFor("");
      setSuggestLoading(false);
      setSuggestUnavailable(false);
      setActiveIndex(-1);
      return;
    }
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setSuggestLoading(true);
    timeoutRef.current = setTimeout(async () => {
      if (abortControllerRef.current) abortControllerRef.current.abort();
      abortControllerRef.current = new AbortController();

      try {
        const results = await fetchGeocode(query, "suggest", lang, abortControllerRef.current.signal);
        setSuggestions(results);
        setSuggestionsFor(query);
        setSuggestUnavailable(false);
        setActiveIndex(-1);
        // Only open if the user is still in the box (e.g. not after a
        // language switch re-ran this effect).
        if (document.activeElement === inputRef.current) setShowSuggestions(true);
        setSuggestLoading(false);
      } catch (err) {
        if (err.name !== 'AbortError') {
          if (!err.status) console.error("Autocomplete error:", err);
          setSuggestions([]);
          setSuggestionsFor("");
          setSuggestUnavailable(true);
          setSuggestLoading(false);
        }
      }
    }, SUGGEST_DEBOUNCE_MS);
    return () => {
      clearTimeout(timeoutRef.current);
      if (abortControllerRef.current) abortControllerRef.current.abort();
    };
  }, [searchInput, searchType, lang]);

  // The "still loading" note is moot once predictions arrive and App selects
  // the pending location.
  useEffect(() => {
    if (predictionsReady) setInfo("");
  }, [predictionsReady]);

  const runSearch = (coords) => {
    const status = onSearch(coords);
    setInfo(status === "pending" ? t(lang, "search.pending") : "");
  };

  // Click-outside
  useEffect(() => {
    const handle = (e) => {
      if (searchRef.current && !searchRef.current.contains(e.target)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, []);

  const geocodeAddress = async (address, lat, lon) => {
    try {
      setIsSearching(true);
      setError("");
      if (lat < 25.8 || lat > 36.5 || lon < -106.6 || lon > -93.5) {
        setError(t(lang, "search.errors.address_outside"));
        setIsSearching(false);
        return;
      }
      runSearch({ lat, lon, address });
      setShowSuggestions(false);
    } catch (err) {
      setError(t(lang, "search.errors.search_failed"));
      console.error(err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    const label = suggestion.name || suggestion.display_name.split(",")[0];
    clearTimeout(timeoutRef.current);
    if (abortControllerRef.current) abortControllerRef.current.abort();
    setSuggestLoading(false);
    if (label !== searchInput) skipNextQueryRef.current = true;
    setSearchInput(label);
    setSuggestions([]);
    setSuggestionsFor("");
    setActiveIndex(-1);
    geocodeAddress(suggestion.display_name, parseFloat(suggestion.lat), parseFloat(suggestion.lon));
  };

  // Keyboard navigation over the dropdown (ArrowUp/Down + Enter + Escape),
  // like every mainstream map search box.
  const handleInputKeyDown = (e) => {
    if (searchType !== "address" || !showSuggestions || suggestions.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => (i + 1) % suggestions.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => (i <= 0 ? suggestions.length - 1 : i - 1));
    } else if (e.key === "Escape") {
      setShowSuggestions(false);
      setActiveIndex(-1);
    } else if (e.key === "Enter" && activeIndex >= 0 && activeIndex < suggestions.length) {
      e.preventDefault();
      handleSuggestionClick(suggestions[activeIndex]);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    setError("");

    if (searchType === "address") {
      const query = searchInput.trim();
      if (!query) {
        setError(t(lang, "search.errors.enter_address"));
        return;
      }
      const fresh = suggestions.length > 0 && suggestionsFor === query;
      if (fresh && activeIndex >= 0 && activeIndex < suggestions.length) {
        handleSuggestionClick(suggestions[activeIndex]);
      } else if (fresh) {
        handleSuggestionClick(suggestions[0]);
      } else {
        // No suggestions for this exact text (instant Enter, or Photon found
        // nothing): one explicit submit query, which may use the Nominatim
        // fallback server-side.
        clearTimeout(timeoutRef.current);
        if (abortControllerRef.current) abortControllerRef.current.abort();
        setSuggestLoading(false);
        setShowSuggestions(false);
        setIsSearching(true);
        try {
          const results = await fetchGeocode(query, "submit", lang);
          if (results.length > 0) {
            const r = results[0];
            geocodeAddress(r.display_name, parseFloat(r.lat), parseFloat(r.lon));
          } else {
            setError(t(lang, "search.errors.address_not_found"));
          }
        } catch (err) {
          setError(t(lang, "search.errors.search_failed"));
        } finally {
          setIsSearching(false);
        }
      }
    } else {
      const coords = searchInput.trim().split(",");
      if (coords.length !== 2) {
        setError(t(lang, "search.errors.coords_format"));
        return;
      }
      const lat = parseFloat(coords[0].trim());
      const lon = parseFloat(coords[1].trim());
      if (isNaN(lat) || isNaN(lon)) {
        setError(t(lang, "search.errors.coords_invalid"));
        return;
      }
      if (lat < 25.8 || lat > 36.5 || lon < -106.6 || lon > -93.5) {
        setError(t(lang, "search.errors.coords_out_of_bounds"));
        return;
      }
      runSearch({ lat, lon });
      setError("");
    }
  };

  return (
    <div className="search-bar" ref={searchRef}>
      <form onSubmit={handleSearch}>
        <div className="search-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={searchType === "address"}
            className={`search-tab ${searchType === "address" ? "active" : ""}`}
            onClick={() => { setSearchType("address"); setSearchInput(""); setSuggestions([]); setError(""); setInfo(""); }}
          >
            <PinIcon />
            <span>{t(lang, "search.address_tab")}</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={searchType === "coordinates"}
            className={`search-tab ${searchType === "coordinates" ? "active" : ""}`}
            onClick={() => { setSearchType("coordinates"); setSearchInput(""); setSuggestions([]); setError(""); setInfo(""); }}
          >
            <CompassIcon />
            <span>{t(lang, "search.coordinates_tab")}</span>
          </button>
        </div>

        <div className="search-input-group">
          <div className="search-input-wrapper">
            <SearchIcon />
            <input
              ref={inputRef}
              type="text"
              maxLength={MAX_QUERY_LEN}
              placeholder={searchType === "address"
                ? t(lang, "search.placeholder_address")
                : t(lang, "search.placeholder_coords")}
              value={searchInput}
              onChange={(e) => { setSearchInput(e.target.value); setInfo(""); }}
              onFocus={() => searchType === "address" && setShowSuggestions(true)}
              onKeyDown={handleInputKeyDown}
              required
              autoComplete="off"
              aria-label={searchType === "address"
                ? t(lang, "search.placeholder_address")
                : t(lang, "search.placeholder_coords")}
            />

            {searchType === "address" && showSuggestions &&
              (suggestions.length > 0 || suggestLoading ||
                (!suggestUnavailable && suggestionsFor === searchInput.trim())) && (
              <div className="suggestions-dropdown" role="listbox">
                {suggestions.map((s, idx) => (
                  <div
                    key={`${s.lat},${s.lon},${idx}`}
                    className={`suggestion-item${idx === activeIndex ? " active" : ""}`}
                    role="option"
                    aria-selected={idx === activeIndex}
                    tabIndex={0}
                    onClick={() => handleSuggestionClick(s)}
                    onMouseEnter={() => setActiveIndex(idx)}
                    onKeyDown={(e) => e.key === "Enter" && handleSuggestionClick(s)}
                  >
                    <div className="suggestion-name">{s.name || s.display_name.split(",")[0]}</div>
                    <div className="suggestion-address">{s.display_name}</div>
                  </div>
                ))}
                {suggestions.length === 0 && (
                  <div className="suggestion-status" aria-live="polite">
                    {suggestLoading
                      ? t(lang, "search.searching")
                      : t(lang, "search.errors.address_not_found")}
                  </div>
                )}
              </div>
            )}
          </div>

          <button type="submit" disabled={isSearching}>
            {isSearching ? t(lang, "search.searching") : t(lang, "search.search_button")}
          </button>
        </div>

        {error && (
          <div className="search-error" role="alert">
            <AlertIcon />
            <span>{error}</span>
          </div>
        )}
        {!error && info && (
          <div className="search-info" role="status">
            <span>{info}</span>
          </div>
        )}
      </form>

      <div className="search-hint">
        {searchType === "coordinates" ? t(lang, "search.hint_coords") : ""}
      </div>
    </div>
  );
}
