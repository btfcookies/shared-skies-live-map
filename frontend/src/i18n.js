// English ordinal suffix: 1st, 2nd, 3rd, 4th ... 11th-13th, 21st, 22nd ...
function enOrdinal(n) {
  const mod100 = n % 100;
  if (mod100 >= 11 && mod100 <= 13) return `${n}th`;
  switch (n % 10) {
    case 1: return `${n}st`;
    case 2: return `${n}nd`;
    case 3: return `${n}rd`;
    default: return `${n}th`;
  }
}

const TRANSLATIONS = {
  en: {
    brand_tagline_all: "All of Texas · Census Tract Level",
    back_to: (name) => `Back to ${name}`,
    current_pm25: "Predicted PM2.5 · 24-hr avg",
    current_conditions: "Current Conditions",
    temperature: "Temperature",
    humidity: "Humidity",
    pressure: "Pressure",
    wind: "Wind",
    use_search: "Use the search bar to find any Texas address and view its air quality. Tap any tract on the map for details with real-time weather.",
    visits_label: (n) => `${n} visits`,
    visits_tooltip: "Total visits since launch",
    updated_minutes_ago: (n) => `Updated ${n} min ago`,
    updated_just_now: "Updated just now",
    refresh_note: (n) => `sensor feed refreshes about every ${n} min`,
    refreshing: "Updating...",
    live_badge_live: "Live sensor data",
    live_badge_fallback: "Live sensor feed unavailable - showing typical conditions",
    predictions_error_title: "Couldn't load air-quality predictions yet.",
    predictions_error_body: "The server may be waking up - this can take up to a minute. Retrying automatically...",
    map_loading_tracts: "Loading census tracts...",
    map_waking_server: "Waking up the server - this can take up to a minute...",
    map_tracts_error: "Couldn't load the tract map. Retrying...",
    legend_toggle_aria: "Show or hide the legend",
    lang_selector_aria: "Language",
    tabs: {
      aria: "Tabs",
      map: "Map",
      sensors: "Sensors",
      guide: "Guide",
    },
    percentile: (n) => `${enOrdinal(n)} percentile`,
    // EJScreen fields are national percentiles (T_* columns, pipeline/01_build_tract_lookup.py).
    ej_fields: {
      ejf_score:           { label: "Demographic Index", desc: "EJScreen Demographic Index (people of color + low income), national percentile" },
      pct_people_of_color: { label: "People of Color",   desc: "National percentile for the share of residents who are people of color" },
      pct_low_income:      { label: "Low Income",        desc: "National percentile for the share of residents below 2x the federal poverty level" },
      traffic_proximity:   { label: "Traffic Proximity", desc: "National percentile for traffic proximity and volume" },
      diesel_pm_proximity: { label: "Diesel PM",         desc: "National percentile for diesel particulate matter in air" },
      superfund_proximity: { label: "Superfund Sites",   desc: "National percentile for proximity to EPA Superfund sites" },
    },
    credits_label: "Data",
    // Each credit is a list of plain-text and linked segments.
    credits: [
      ["Sensor data: ", { text: "PurpleAir", href: "https://www.purpleair.com/" }],
      ["Weather: ", { text: "Open-Meteo.com", href: "https://open-meteo.com/" }, " (", { text: "CC BY 4.0", href: "https://creativecommons.org/licenses/by/4.0/" }, ")"],
      ["Aerosols: contains modified ", { text: "Copernicus Atmosphere Monitoring Service", href: "https://atmosphere.copernicus.eu/" }, " information"],
      ["Smoke: ", { text: "NOAA HMS", href: "https://www.ospo.noaa.gov/products/land/hms.html" }],
      ["Tracts: ", { text: "U.S. Census TIGER", href: "https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html" }],
      [{ text: "EPA EJScreen", href: "https://www.epa.gov/ejscreen" }, " (context only)"],
    ],
    ej_context: "Environmental Justice Context",
    tract_distribution: "Tract Distribution",
    loading_predictions: "Generating predictions for all Texas tracts...",
    // Air Quality Guide
    guide_intro: "PM2.5 refers to fine particulate matter, tiny particles 2.5 micrometers or smaller. They can penetrate deep into the lungs and enter the bloodstream.",
    guide: {
      why_title: "Why PM2.5 Matters",
      why_body: "Particles this small bypass the nose and throat and lodge deep in lung tissue. Short-term exposure causes coughing and irritation. Long-term exposure is linked to heart disease, stroke, lung cancer, and reduced life expectancy.",
      about_title: "About These Predictions",
      about_body: "Values combine live PurpleAir sensor readings, weather data, and NOAA satellite smoke analysis. No demographic or environmental-justice variable is used as a model input. The PurpleAir feed is refreshed about every 15 minutes and predictions are recomputed after each refresh; weather and aerosol inputs update every 6 hours and smoke analysis every hour. Values are estimated 24-hour-average ground-level concentrations for each census tract.",
      levels: {
        good: {
          name: "Good",
          range: "0 – 9 µg/m³",
          description: "Within the U.S. EPA annual PM2.5 standard (9 µg/m³). No health concerns for anyone — ideal for outdoor activities.",
          who: "Safe for everyone",
        },
        moderate: {
          name: "Moderate",
          range: "9 – 13 µg/m³",
          description: "Just above the U.S. EPA annual standard (9 µg/m³). Most people experience no effects; a few unusually sensitive individuals may notice minor symptoms.",
          who: "Unusually sensitive people: consider reducing prolonged outdoor exertion",
        },
        elevated: {
          name: "Elevated",
          range: "13 – 17 µg/m³",
          description: "Above the WHO 24-hour guideline (15 µg/m³). Prolonged exposure carries cardiovascular and respiratory risk; common in dense urban/industrial areas and during smoke or dust events.",
          who: "Sensitive groups (asthma, heart disease, elderly, children): limit prolonged outdoor exertion.",
        },
        high: {
          name: "High",
          range: "17+ µg/m³",
          description: "Everyone may begin to experience effects; sensitive groups (asthma, heart disease, elderly, children) are at greater risk. Usually driven by wildfire smoke or dust.",
          who: "Everyone: limit prolonged or heavy outdoor activity. Sensitive groups: stay indoors when possible.",
        },
      },
    },
    // Search / UI
    search: {
      address_tab: "Address",
      coordinates_tab: "Coordinates",
      placeholder_address: "Enter address (e.g., Austin, TX)",
      placeholder_coords: "Latitude, Longitude (e.g., 30.2672, -97.7431)",
      search_button: "Search",
      searching: "Searching...",
      loading: "Loading...",
      open: "Open search",
      close: "Close search",
      pending: "Predictions are still loading - this location will be selected as soon as they're ready.",
      errors: {
        address_outside: "Address is outside Texas. Please enter a Texas address.",
        enter_address: "Please enter an address",
        address_not_found: "Address not found. Try a more specific Texas address.",
        search_failed: "Search failed. Please try again.",
        coords_format: "Please enter coordinates as: latitude, longitude",
        coords_invalid: "Please enter valid numbers for latitude and longitude",
        coords_out_of_bounds: "Coordinates must be within Texas bounds",
      },
      hint_coords: "Enter latitude and longitude separated by comma",
    },
    tooltip: {
      county_suffix: "County",
      updating: "Updating...",
      aqi_label: "AQI",
      modeled_note: "Model prediction · 24-hr average",
    },
    aqi_equiv: (aqi) => `≈ US EPA AQI ${aqi}`,
    live_sensors_note: (n) => `${n} PurpleAir sensors`,
    legend_title: "PM2.5 µg/m³",
    census_tract_prefix: "Census Tract",
    // Quantum Sensor Placement
    quantum: {
      tab: "Sensor Placement",
      title: "Quantum-Inspired Sensor Placement",
      subtitle: "Finding locations for new air-quality sensors with quantum-inspired optimization (a QUBO solved by classical simulated annealing)",
      loading: "Running the placement optimizer...",
      loading_detail: "Solving a QUBO with classical simulated annealing across 6,896 tracts",
      loading_slow: "Still working - the first run after the server wakes up can take a minute or two.",
      error: "Couldn't load sensor-placement results. The server may be busy or waking up.",
      retry: "Try again",
      method: "Method",
      weights: "Objective weights",
      weight_names: {
        coverage: "Coverage gap",
        error: "Model error",
        ej: "Demographic Index",
        pm25: "Predicted PM2.5",
      },
      feasible_yes: (k) => `Best solution satisfies the ${k}-site constraint`,
      feasible_no: (n, k) => `Only ${n} well-spaced sites could be placed (target ${k})`,
      candidate_pool: (n) => `Candidate pool: ${n} tracts`,
      existing_sensors: (n) => `${n} existing PurpleAir sensors`,
      existing_hint: "gray dots on the map",
      new_placements: "new placements",
      new_suffix: "new",
      view_tract: (id, county) => `View tract ${id} ${county}`,
      recommended_sensor: "Recommended Sensor",
      ej_abbr: "EJ",
      ej_abbr_title: "EJ priority (EJScreen Demographic Index)",
      cov_abbr: "Cov",
      cov_abbr_title: "Coverage need (distance from existing sensors)",
      num_sensors: "Sensors to Place",
      coverage: "Coverage",
      avg_distance: "Avg Distance to Sensor",
      max_gap: "Largest Gap",
      ej_equity: "Demographic Index Equity",
      method_comparison: "Method Comparison",
      quantum_annealing: "QUBO + Simulated Annealing",
      greedy: "Greedy Algorithm",
      classical_sa: "Classical Sim. Annealing",
      tracts_covered: "Tracts Covered",
      avg_ej: "Avg Demographic Index",
      runtime: "Runtime",
      coverage_by_ej: "Coverage by Demographic Index Quartile",
      q1_low: "Q1 (Lowest)",
      q2: "Q2",
      q3: "Q3",
      q4_high: "Q4 (Highest)",
      recommended_locations: "Recommended Sensor Locations",
      rank: "Rank",
      tract: "Tract",
      county: "County",
      composite: "Score",
      view_on_map: "View on Map",
      tab_intro_title: "About this tab",
      tab_intro_body: ({ k, n, existing }) => `This tab proposes ${k} locations for new PM2.5 sensors across Texas. For each candidate tract, the optimizer weighs four signals: distance from ${existing ? `the ${existing} existing PurpleAir sensors it was given (gray dots)` : "the existing PurpleAir network (gray dots)"}, the model's estimated error (interpolated from how well it predicts nearby sensors), predicted PM2.5, and the EPA EJScreen Demographic Index (people of color and low-income residents). Only tracts at least 8 miles from an existing sensor can be picked, and water tracts and tracts with no EJScreen score are left out, so most metro neighborhoods, which already have sensors nearby, are not candidates. Each blue dot is a recommended placement. Click one to inspect that tract.`,
      what_is: "What is quantum-inspired optimization?",
      what_is_body: ({ k, n, total, candidates, candidateTotal }) => `Choosing ${k} sites from ${n} census tracts allows about ${total} possible arrangements, far too many to check one by one. The optimizer first narrows the field to ${candidates ? `${candidates} geographically spread-out candidate tracts (about ${candidateTotal} possible ${k}-site sets)` : "a smaller pool of geographically spread-out candidate tracts"}. It then writes the choice as a QUBO (quadratic unconstrained binary optimization) problem and solves it with classical simulated annealing on an ordinary server. A QUBO is the same problem format that quantum annealers accept, which is why this is called quantum-inspired, but no quantum hardware is used. The annealer draws many low-energy samples, and a swap-based local search (trading one site for another while that lowers the energy) then refines the best of them into the final recommendations.`,
      why_ej: "Why weight placement by demographics?",
      why_ej_body: ({ coverage, error, ej, pm25 }) => `PM2.5 exposure is not evenly distributed: communities of color and low-income neighborhoods often face higher pollution with fewer monitors nearby. The optimizer scores each candidate on four signals: distance from existing sensors (${coverage}% of the weight), the model's estimated error (${error}%), the EJScreen Demographic Index, which combines the share of people of color and of low-income residents (${ej}%), and predicted PM2.5 (${pm25}%). Among the eligible tracts, this steers new sensors toward communities that face more of that burden instead of only filling the gaps that are easiest to cover.`,
      show_quantum: "Quantum",
      show_greedy: "Greedy",
      show_classical: "Classical",
      generated: (when) => `Generated ${when}`,
    },
  },
  es: {
    brand_tagline_all: "Todo Texas · Nivel de tracto censal",
    back_to: (name) => `Volver a ${name}`,
    current_pm25: "PM2.5 previsto · media 24 h",
    current_conditions: "Condiciones actuales",
    temperature: "Temperatura",
    humidity: "Humedad",
    pressure: "Presión",
    wind: "Viento",
    use_search: "Usa la barra de búsqueda para encontrar cualquier dirección en Texas y ver su calidad del aire. Toca cualquier tracto en el mapa para detalles con el clima en tiempo real.",
    visits_label: (n) => `${n} visitas`,
    visits_tooltip: "Visitas totales desde el lanzamiento",
    updated_minutes_ago: (n) => `Actualizado hace ${n} min`,
    updated_just_now: "Actualizado ahora mismo",
    refresh_note: (n) => `los sensores se actualizan cada ~${n} min`,
    refreshing: "Actualizando...",
    live_badge_live: "Datos de sensores en vivo",
    live_badge_fallback: "Datos de sensores en vivo no disponibles - mostrando condiciones típicas",
    predictions_error_title: "Aún no se pudieron cargar las predicciones de calidad del aire.",
    predictions_error_body: "Es posible que el servidor se esté activando; esto puede tardar hasta un minuto. Reintentando automáticamente...",
    map_loading_tracts: "Cargando tractos censales...",
    map_waking_server: "Activando el servidor; esto puede tardar hasta un minuto...",
    map_tracts_error: "No se pudo cargar el mapa de tractos. Reintentando...",
    legend_toggle_aria: "Mostrar u ocultar la leyenda",
    lang_selector_aria: "Idioma",
    tabs: {
      aria: "Pestañas",
      map: "Mapa",
      sensors: "Sensores",
      guide: "Guía",
    },
    percentile: (n) => `Percentil ${n}`,
    ej_fields: {
      ejf_score:           { label: "Índice demográfico", desc: "Índice Demográfico de EJScreen (personas de color + bajos ingresos), percentil nacional" },
      pct_people_of_color: { label: "Personas de color",  desc: "Percentil nacional de la proporción de residentes que son personas de color" },
      pct_low_income:      { label: "Bajos ingresos",     desc: "Percentil nacional de la proporción de residentes por debajo de 2 veces el umbral federal de pobreza" },
      traffic_proximity:   { label: "Tráfico",            desc: "Percentil nacional de cercanía y volumen de tráfico" },
      diesel_pm_proximity: { label: "PM diésel",          desc: "Percentil nacional de partículas de diésel en el aire" },
      superfund_proximity: { label: "Sitios Superfund",   desc: "Percentil nacional de cercanía a sitios Superfund de la EPA" },
    },
    credits_label: "Datos",
    credits: [
      ["Datos de sensores: ", { text: "PurpleAir", href: "https://www.purpleair.com/" }],
      ["Clima: ", { text: "Open-Meteo.com", href: "https://open-meteo.com/" }, " (", { text: "CC BY 4.0", href: "https://creativecommons.org/licenses/by/4.0/" }, ")"],
      ["Aerosoles: contiene información modificada del ", { text: "Servicio de Vigilancia Atmosférica de Copernicus", href: "https://atmosphere.copernicus.eu/" }],
      ["Humo: ", { text: "NOAA HMS", href: "https://www.ospo.noaa.gov/products/land/hms.html" }],
      ["Tractos: ", { text: "TIGER de la Oficina del Censo de EE. UU.", href: "https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html" }],
      [{ text: "EPA EJScreen", href: "https://www.epa.gov/ejscreen" }, " (solo contexto)"],
    ],
    ej_context: "Contexto de Justicia Ambiental",
    tract_distribution: "Distribución de tractos",
    loading_predictions: "Generando predicciones para todos los tractos de Texas...",
    // Air Quality Guide (Spanish)
    guide_intro: "PM2.5 se refiere a las partículas finas, diminutas partículas de 2.5 micrómetros o menos. Pueden penetrar profundamente en los pulmones y entrar al torrente sanguíneo.",
    guide: {
      why_title: "Por qué importa PM2.5",
      why_body: "Partículas tan pequeñas evitan la nariz y la garganta y se alojan en el tejido pulmonar. La exposición a corto plazo causa tos e irritación. La exposición a largo plazo se relaciona con enfermedades cardíacas, accidente cerebrovascular, cáncer de pulmón y menor esperanza de vida.",
      about_title: "Sobre estas predicciones",
      about_body: "Los valores combinan lecturas en vivo de sensores PurpleAir, datos meteorológicos y análisis satelital de humo de la NOAA. Ninguna variable demográfica o de justicia ambiental se usa como entrada del modelo. Los datos de PurpleAir se actualizan aproximadamente cada 15 minutos y las predicciones se recalculan después de cada actualización; los datos meteorológicos y de aerosoles se actualizan cada 6 horas y el análisis de humo cada hora. Los valores son concentraciones estimadas a nivel del suelo, promediadas en 24 horas, para cada tracto censal.",
      levels: {
        good: {
          name: "Bueno",
          range: "0 – 9 µg/m³",
          description: "Dentro del estándar anual de PM2.5 de la EPA de EE. UU. (9 µg/m³). Sin preocupaciones de salud para nadie — ideal para actividades al aire libre.",
          who: "Seguro para todos",
        },
        moderate: {
          name: "Moderado",
          range: "9 – 13 µg/m³",
          description: "Apenas por encima del estándar anual de la EPA de EE. UU. (9 µg/m³). La mayoría no experimenta efectos; algunos individuos inusualmente sensibles pueden notar síntomas menores.",
          who: "Personas sensibles: considere reducir el esfuerzo prolongado al aire libre",
        },
        elevated: {
          name: "Elevado",
          range: "13 – 17 µg/m³",
          description: "Por encima de la guía de 24 horas de la OMS (15 µg/m³). La exposición prolongada conlleva riesgo cardiovascular y respiratorio; común en zonas urbanas/industriales densas y durante eventos de humo o polvo.",
          who: "Grupos sensibles (asma, enfermedades cardíacas, ancianos, niños): limitar el esfuerzo prolongado al aire libre.",
        },
        high: {
          name: "Alto",
          range: "17+ µg/m³",
          description: "Todos pueden comenzar a experimentar efectos; los grupos sensibles (asma, enfermedades cardíacas, ancianos, niños) tienen mayor riesgo. Generalmente por humo de incendios o polvo.",
          who: "Todos: limitar la actividad al aire libre prolongada o intensa. Grupos sensibles: permanecer en interiores cuando sea posible.",
        },
      },
    },
    // Search / UI (Spanish)
    search: {
      address_tab: "Dirección",
      coordinates_tab: "Coordenadas",
      placeholder_address: "Introduce dirección (p. ej., Austin, TX)",
      placeholder_coords: "Latitud, Longitud (p. ej., 30.2672, -97.7431)",
      search_button: "Buscar",
      searching: "Buscando...",
      loading: "Cargando...",
      open: "Abrir búsqueda",
      close: "Cerrar búsqueda",
      pending: "Las predicciones aún se están cargando; esta ubicación se seleccionará en cuanto estén listas.",
      errors: {
        address_outside: "La dirección está fuera de Texas. Por favor ingresa una dirección en Texas.",
        enter_address: "Por favor ingresa una dirección",
        address_not_found: "Dirección no encontrada. Intenta una dirección más específica en Texas.",
        search_failed: "La búsqueda falló. Por favor inténtalo de nuevo.",
        coords_format: "Por favor ingresa coordenadas como: latitud, longitud",
        coords_invalid: "Por favor ingresa números válidos para latitud y longitud",
        coords_out_of_bounds: "Las coordenadas deben estar dentro de Texas",
      },
      hint_coords: "Introduce latitud y longitud separadas por coma",
    },
    tooltip: {
      county_suffix: "Condado",
      updating: "Actualizando...",
      aqi_label: "AQI",
      modeled_note: "Predicción del modelo · media de 24 h",
    },
    aqi_equiv: (aqi) => `≈ AQI ${aqi} (EPA de EE. UU.)`,
    live_sensors_note: (n) => `${n} sensores PurpleAir`,
    legend_title: "PM2.5 µg/m³",
    census_tract_prefix: "Tracto censal",
    // Quantum Sensor Placement (Spanish)
    quantum: {
      tab: "Ubicación de Sensores",
      title: "Ubicación de Sensores de Inspiración Cuántica",
      subtitle: "Buscando ubicaciones para nuevos sensores de calidad del aire con optimización de inspiración cuántica (un QUBO resuelto con recocido simulado clásico)",
      loading: "Ejecutando el optimizador de ubicaciones...",
      loading_detail: "Resolviendo un QUBO con recocido simulado clásico en 6,896 tractos",
      loading_slow: "Seguimos trabajando; la primera ejecución después de que el servidor se activa puede tardar uno o dos minutos.",
      error: "No se pudieron cargar los resultados de ubicación de sensores. Es posible que el servidor esté ocupado o activándose.",
      retry: "Intentar de nuevo",
      method: "Método",
      weights: "Pesos del objetivo",
      weight_names: {
        coverage: "Brecha de cobertura",
        error: "Error del modelo",
        ej: "Índice demográfico",
        pm25: "PM2.5 previsto",
      },
      feasible_yes: (k) => `La mejor solución cumple la restricción de ${k} sitios`,
      feasible_no: (n, k) => `Solo se pudieron ubicar ${n} sitios bien espaciados (objetivo ${k})`,
      candidate_pool: (n) => `Grupo de candidatos: ${n} tractos`,
      existing_sensors: (n) => `${n} sensores PurpleAir existentes`,
      existing_hint: "puntos grises en el mapa",
      new_placements: "nuevos a colocar",
      new_suffix: "nuevos",
      view_tract: (id, county) => `Ver tracto ${id} ${county}`,
      recommended_sensor: "Sensor Recomendado",
      ej_abbr: "JA",
      ej_abbr_title: "Prioridad de justicia ambiental (Índice Demográfico de EJScreen)",
      cov_abbr: "Cob",
      cov_abbr_title: "Necesidad de cobertura (distancia a los sensores existentes)",
      num_sensors: "Sensores a Colocar",
      coverage: "Cobertura",
      avg_distance: "Distancia Prom. al Sensor",
      max_gap: "Mayor Brecha",
      ej_equity: "Equidad por Índice Demográfico",
      method_comparison: "Comparación de Métodos",
      quantum_annealing: "QUBO + Recocido Simulado",
      greedy: "Algoritmo Voraz",
      classical_sa: "Recocido Simulado Clásico",
      tracts_covered: "Tractos Cubiertos",
      avg_ej: "Índice Demográfico Prom.",
      runtime: "Tiempo de Ejecución",
      coverage_by_ej: "Cobertura por Cuartil del Índice Demográfico",
      q1_low: "Q1 (Más bajo)",
      q2: "Q2",
      q3: "Q3",
      q4_high: "Q4 (Más alto)",
      recommended_locations: "Ubicaciones Recomendadas",
      rank: "Rango",
      tract: "Tracto",
      county: "Condado",
      composite: "Puntaje",
      view_on_map: "Ver en Mapa",
      tab_intro_title: "Sobre esta pestaña",
      tab_intro_body: ({ k, n, existing }) => `Esta pestaña propone ${k} ubicaciones para nuevos sensores PM2.5 en Texas. Para cada tracto candidato, el optimizador pondera cuatro señales: la distancia a ${existing ? `los ${existing} sensores PurpleAir existentes que recibió (puntos grises)` : "la red existente de sensores PurpleAir (puntos grises)"}, el error estimado del modelo (interpolado a partir de qué tan bien predice los sensores cercanos), el PM2.5 previsto y el Índice Demográfico de EJScreen de la EPA (personas de color y residentes de bajos ingresos). Solo pueden elegirse tractos a al menos 8 millas de un sensor existente, y se excluyen los tractos de agua y los que no tienen puntaje de EJScreen, por lo que la mayoría de los vecindarios metropolitanos, que ya tienen sensores cerca, no son candidatos. Cada punto azul es una ubicación recomendada. Haz clic en uno para inspeccionar ese tracto.`,
      what_is: "¿Qué es la optimización de inspiración cuántica?",
      what_is_body: ({ k, n, total, candidates, candidateTotal }) => `Elegir ${k} sitios entre ${n} tractos censales permite unas ${total} combinaciones posibles, demasiadas para revisarlas una por una. El optimizador primero reduce el campo a ${candidates ? `${candidates} tractos candidatos repartidos geográficamente (unos ${candidateTotal} conjuntos posibles de ${k} sitios)` : "un grupo más pequeño de tractos candidatos repartidos geográficamente"}. Luego plantea la elección como un problema QUBO (optimización binaria cuadrática sin restricciones) y lo resuelve con recocido simulado clásico en un servidor común. Un QUBO es el mismo formato de problema que aceptan los recocedores cuánticos, por eso se llama de inspiración cuántica, pero aquí no se usa hardware cuántico. El recocido genera muchas muestras de baja energía y luego una búsqueda local por intercambios (cambiar un sitio por otro mientras eso reduzca la energía) refina las mejores hasta obtener las recomendaciones finales.`,
      why_ej: "¿Por qué ponderar la ubicación según la demografía?",
      why_ej_body: ({ coverage, error, ej, pm25 }) => `La exposición a PM2.5 no se distribuye uniformemente: las comunidades de color y los vecindarios de bajos ingresos suelen enfrentar más contaminación con menos monitores cerca. El optimizador califica cada candidato con cuatro señales: la distancia a los sensores existentes (${coverage}% del peso), el error estimado del modelo (${error}%), el Índice Demográfico de EJScreen, que combina la proporción de personas de color y de residentes de bajos ingresos (${ej}%), y el PM2.5 previsto (${pm25}%). Entre los tractos elegibles, así se dirigen los nuevos sensores hacia las comunidades que enfrentan más esa carga, en lugar de solo llenar los vacíos más fáciles de cubrir.`,
      show_quantum: "Cuántico",
      show_greedy: "Voraz",
      show_classical: "Clásico",
      generated: (when) => `Generado ${when}`,
    },
  }
};

const CATEGORY_MAP = {
  en: {
    Good: "Good",
    Moderate: "Moderate",
    Elevated: "Elevated",
    High: "High",
  },
  es: {
    Good: "Bueno",
    Moderate: "Moderado",
    Elevated: "Elevado",
    High: "Alto",
  }
};

const HEALTH_MSG = {
  en: {
    Good: "Air quality is good — within the U.S. EPA annual PM2.5 standard (9 µg/m³).",
    Moderate: "Moderate — above the EPA annual standard. Unusually sensitive people may want to limit prolonged outdoor exertion.",
    Elevated: "Elevated — above the WHO 24-hour guideline (15 µg/m³). Sensitive groups should limit prolonged outdoor activity.",
    High: "⚠️ High — everyone may begin to feel effects; sensitive groups are at greater risk. Often driven by wildfire smoke or dust.",
  },
  es: {
    Good: "La calidad del aire es buena — dentro del estándar anual de PM2.5 de la EPA de EE. UU. (9 µg/m³).",
    Moderate: "Moderado — por encima del estándar anual de la EPA. Las personas inusualmente sensibles podrían limitar el esfuerzo prolongado al aire libre.",
    Elevated: "Elevado — por encima de la guía de 24 horas de la OMS (15 µg/m³). Los grupos sensibles deben limitar la actividad prolongada al aire libre.",
    High: "⚠️ Alto — todos pueden empezar a sentir efectos; los grupos sensibles tienen mayor riesgo. A menudo por humo de incendios o polvo.",
  }
};

export function t(lang = 'en', key, ...args) {
  const group = TRANSLATIONS[lang] || TRANSLATIONS.en;
  const val = group[key];
  if (typeof val === 'function') return val(...args);
  // support nested keys like 'guide.levels.good.name'
  if (val == null && key.includes('.')) {
    const parts = key.split('.');
    let cur = group;
    for (const p of parts) {
      cur = cur?.[p];
      if (cur == null) break;
    }
    if (typeof cur === 'function') return cur(...args);
    if (cur != null) return cur;
    // fallback to english
    cur = TRANSLATIONS.en;
    for (const p of parts) {
      cur = cur?.[p];
      if (cur == null) break;
    }
    if (typeof cur === 'function') return cur(...args);
    if (cur != null) return cur;
  }
  if (typeof val === 'function') return val(...args);
  return val ?? TRANSLATIONS.en[key] ?? key;
}

export function translateCategory(lang = 'en', cat) {
  return CATEGORY_MAP[lang]?.[cat] ?? cat;
}

export function translateHealth(lang = 'en', cat) {
  return HEALTH_MSG[lang]?.[cat] ?? cat;
}

export function formatPercentile(lang = 'en', val) {
  if (val == null || isNaN(val)) return "—";
  return t(lang, "percentile", Math.round(val));
}

export default { t, translateCategory, translateHealth, formatPercentile };
