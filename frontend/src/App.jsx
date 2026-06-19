import { useEffect, useState, useRef, useMemo } from "react";
import L from "leaflet";
import {
  MapContainer, TileLayer, CircleMarker, Popup,
  Rectangle, useMapEvents, useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API = "";

const CLUSTER_COLORS = {
  0: "#22c55e", 1: "#3b82f6", 2: "#f59e0b", 3: "#ef4444",
  4: "#8b5cf6", 5: "#06b6d4", 6: "#111827", "-1": "#6b7280",
};
const CLUSTER_LABELS = {
  0: "High Performers", 1: "Average", 2: "Wake-Affected",
  3: "Underperformers", 4: "Group 4", 5: "Group 5", 6: "Faulty",
};

function modisUrl() {
  const d = new Date(Date.now() - 86_400_000).toISOString().slice(0, 10);
  return `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/${d}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`;
}

// ── LCOE helper ───────────────────────────────────────────────────────────────
function calcLCOE(capacityMw, capacityFactor, priceEurMwh) {
  const LIFETIME = 20, CAPEX_MW = 1_300_000, OPEX_MW_YR = 40_000, R = 0.05;
  const capex     = capacityMw * CAPEX_MW;
  const pvf       = (1 - Math.pow(1 + R, -LIFETIME)) / R;
  const totalCost = capex + capacityMw * OPEX_MW_YR * pvf;
  const annMwh    = capacityMw * capacityFactor * 8760;
  const lcoe      = totalCost / (annMwh * LIFETIME);
  const annRev    = annMwh * priceEurMwh;
  const annNet    = annRev - capacityMw * OPEX_MW_YR;
  const payback   = annNet > 0 ? capex / annNet : 99;
  let lo = -0.5, hi = 3.0;
  for (let i = 0; i < 60; i++) {
    const m   = (lo + hi) / 2;
    const npv = -capex + annNet * (Math.abs(m) > 1e-6 ? (1 - Math.pow(1 + m, -LIFETIME)) / m : LIFETIME);
    npv > 0 ? (lo = m) : (hi = m);
    if (hi - lo < 1e-4) break;
  }
  return {
    lcoe:    +lcoe.toFixed(1),
    payback: +payback.toFixed(1),
    irr:     +(((lo + hi) / 2) * 100).toFixed(1),
    annRevEur:  Math.round(annRev),
    capexEur:   Math.round(capex),
    co2TonYr:   Math.round(annMwh * 0.4),
    annMwh:     Math.round(annMwh),
  };
}

// ── Radar: pre-create all frames as native Leaflet layers, animate via opacity ─
// Using map.removeLayer / map.addLayer (not l.remove) avoids stale-reference bugs.
function RadarLayer({ paths, frameIdx, show, opacity = 0.6 }) {
  const map        = useMap();
  const layersRef  = useRef([]);
  const frameRef   = useRef(frameIdx);
  const showRef    = useRef(show);
  const opacityRef = useRef(opacity);

  useEffect(() => { frameRef.current  = frameIdx;  }, [frameIdx]);
  useEffect(() => { showRef.current   = show;       }, [show]);
  useEffect(() => { opacityRef.current = opacity;   }, [opacity]);

  // Rebuild layers only when the path list changes
  useEffect(() => {
    layersRef.current.forEach(l => { try { map.removeLayer(l); } catch {} });
    layersRef.current = [];
    if (!paths.length || !map) return;

    const newLayers = paths.map((path, i) =>
      L.tileLayer(`${path}/256/{z}/{x}/{y}/4/1_1.png`, {
        opacity: showRef.current && i === frameRef.current ? opacityRef.current : 0,
        zIndex:  500,
        maxZoom: 18,
      }).addTo(map)
    );
    layersRef.current = newLayers;

    return () => {
      newLayers.forEach(l => { try { map.removeLayer(l); } catch {} });
      layersRef.current = [];
    };
  // paths.join is the stable dep; map never changes identity in react-leaflet
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paths.join(","), map]);

  // On every frame tick, flip opacity (no tile reload)
  useEffect(() => {
    layersRef.current.forEach((l, i) => {
      try { l.setOpacity(show && i === frameIdx ? opacity : 0); } catch {}
    });
  }, [frameIdx, show, opacity]);

  return null;
}

// ── Monthly performance + forecast chart ──────────────────────────────────────
// mode="scada"  → has real actual_kw + fleet predicted_kw (Chinese turbines)
// mode="era5"   → ERA5 modelled estimate only, no real measurements (German sites)
function MonthlyChart({ data, accent = "#38bdf8", mode = "scada" }) {
  if (!data?.months?.length) return <div className="chart-loading">Loading chart…</div>;

  const W = 500, H = 190;
  const PAD = { top: 24, right: 16, bottom: 36, left: 54 };
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top  - PAD.bottom;

  const months = data.months;

  // For ERA5 mode: estimated_kw is the only line — treat it as the primary (filled) series
  // For SCADA mode: actual_kw is primary (filled), predicted_kw is secondary (dashed)
  const primaryKey   = mode === "era5" ? "estimated_kw" : "actual_kw";
  const secondaryKey = mode === "era5" ? null            : "predicted_kw";

  const maxKw = Math.max(1,
    ...months.map(m => Math.max(m[primaryKey] ?? 0, m[secondaryKey] ?? 0))
  ) * 1.15;

  const xs = i  => PAD.left + (i / Math.max(months.length - 1, 1)) * pw;
  const ys = kw => PAD.top  + ph - Math.min(kw / maxKw, 1) * ph;

  // Split into history and forecast for primary series
  const histPts  = months.map((m, i) => !m.is_forecast && m[primaryKey] != null ? [xs(i), ys(m[primaryKey])] : null).filter(Boolean);
  const fcstPts  = months.map((m, i) =>  m.is_forecast && m[primaryKey] != null ? [xs(i), ys(m[primaryKey])] : null).filter(Boolean);
  const secondPts = secondaryKey
    ? months.filter(m => !m.is_forecast).map((m, i) => {
        const realIdx = months.indexOf(m);
        return [xs(realIdx), ys(m[secondaryKey] ?? 0)];
      })
    : [];

  const pathStr  = pts => pts.length ? "M" + pts.map(([x,y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" L") : "";
  const areaStr  = pts => {
    if (pts.length < 2) return "";
    return `${pathStr(pts)} L${pts[pts.length-1][0].toFixed(1)},${(PAD.top+ph).toFixed(1)} L${pts[0][0].toFixed(1)},${(PAD.top+ph).toFixed(1)} Z`;
  };

  const fcstStart = months.findIndex(m => m.is_forecast);
  const yticks    = [0, 0.25, 0.5, 0.75, 1].map(f => maxKw * f);
  const gradId    = `grad-${accent.replace("#", "")}`;
  const gradFcst  = `gradf-${accent.replace("#", "")}`;

  return (
    <div className="chart-wrap">
      <div className="chart-legend">
        {mode === "era5" ? (
          <>
            <span className="cleg actual" style={{ "--cleg-color": accent }}>— ERA5 estimate</span>
            <span className="cleg forecast">– – Seasonal forecast</span>
            <span style={{ fontSize: 10, color: "#475569", marginLeft: 8 }}>No measured data available</span>
          </>
        ) : (
          <>
            <span className="cleg actual" style={{ "--cleg-color": accent }}>— Actual (SCADA)</span>
            {secondaryKey && <span className="cleg predicted">– – Fleet expected</span>}
            {fcstStart >= 0 && <span className="cleg forecast">▨ Forecast</span>}
          </>
        )}
      </div>

      <svg width={W} height={H} className="perf-chart">
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor={accent} stopOpacity="0.38" />
            <stop offset="100%" stopColor={accent} stopOpacity="0.03" />
          </linearGradient>
          <linearGradient id={gradFcst} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#f59e0b" stopOpacity="0.22" />
            <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.02" />
          </linearGradient>
        </defs>

        {/* Forecast zone background */}
        {fcstStart >= 0 && (
          <rect x={xs(fcstStart) - 1} y={PAD.top} width={W - PAD.right - xs(fcstStart) + 1} height={ph}
            fill="#f59e0b08" stroke="#f59e0b33" strokeWidth="1" strokeDasharray="4,3" />
        )}
        {fcstStart >= 0 && (
          <text x={xs(fcstStart) + 4} y={PAD.top + 10} fill="#f59e0b66" fontSize={8} fontStyle="italic">forecast</text>
        )}

        {/* Y grid + labels */}
        {yticks.map((kw, idx) => (
          <g key={idx}>
            <line x1={PAD.left} x2={W - PAD.right} y1={ys(kw)} y2={ys(kw)} stroke="#1e293b" strokeWidth={1} />
            <text x={PAD.left - 5} y={ys(kw)} textAnchor="end" dominantBaseline="middle" fill="#475569" fontSize={9}>
              {kw >= 1000 ? `${(kw/1000).toFixed(1)}M` : Math.round(kw)}
            </text>
          </g>
        ))}

        {/* X month labels */}
        {months.map((m, i) => (
          <text key={i} x={xs(i)} y={H - 4} textAnchor="middle"
            fill={m.is_forecast ? "#f59e0b77" : "#64748b"} fontSize={8.5}>
            {m.month_label}
          </text>
        ))}

        {/* Fleet expected (SCADA mode only, dashed, history span) */}
        {secondPts.length > 1 && (
          <path d={pathStr(secondPts)} fill="none" stroke="#f59e0b" strokeWidth={1.4} strokeDasharray="5,3" opacity={0.7} />
        )}

        {/* Primary: history area + line */}
        {histPts.length > 1 && <path d={areaStr(histPts)} fill={`url(#${gradId})`} />}
        {histPts.length > 1 && <path d={pathStr(histPts)} fill="none" stroke={accent} strokeWidth={2.2} />}
        {histPts.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={3.5} fill={accent} stroke="#0f172a" strokeWidth={1} />
        ))}

        {/* Forecast: area + dashed line */}
        {fcstPts.length > 0 && histPts.length > 0 && (() => {
          // Connect last history point to first forecast point
          const bridge = [histPts[histPts.length - 1], ...fcstPts];
          return <>
            <path d={areaStr(fcstPts)} fill={`url(#${gradFcst})`} />
            <path d={pathStr(bridge)} fill="none" stroke="#f59e0b" strokeWidth={1.8} strokeDasharray="6,3" opacity={0.85} />
            {fcstPts.map(([x, y], i) => (
              <circle key={i} cx={x} cy={y} r={3} fill="#f59e0b" stroke="#0f172a" strokeWidth={1} />
            ))}
          </>;
        })()}

        {/* Y axis label */}
        <text x={12} y={PAD.top + ph / 2} textAnchor="middle" dominantBaseline="middle"
          fill="#475569" fontSize={9} transform={`rotate(-90, 12, ${PAD.top + ph / 2})`}>kW</text>
      </svg>

      <div className="chart-stats">
        {data.stats && <>
          <div className="cs"><span>Avg actual</span><strong>{data.stats.mean_actual_kw} kW</strong></div>
          <div className="cs"><span>Fleet expected</span><strong>{data.stats.mean_predicted_kw} kW</strong></div>
          <div className="cs"><span>Avg wind</span><strong>{data.stats.mean_wind_ms} m/s</strong></div>
          <div className="cs"><span>Perf. ratio</span>
            <strong className={data.stats.performance_ratio >= 95 ? "good" : data.stats.performance_ratio >= 75 ? "warn" : "bad"}>
              {data.stats.performance_ratio}%
            </strong>
          </div>
        </>}
        {data.annual && <>
          <div className="cs"><span>Avg wind</span><strong>{data.annual.avg_wind_ms} m/s</strong></div>
          <div className="cs"><span>Capacity factor</span><strong>{(+(data.annual.wind_capacity_factor ?? data.annual.solar_capacity_factor ?? 0) * 100).toFixed(1)}%</strong></div>
          <div className="cs"><span>Best month</span><strong>{data.annual.best_wind_month ?? data.annual.best_solar_month}</strong></div>
          <div className="cs"><span>Source</span><strong style={{ color: "#64748b", fontSize: 10 }}>ERA5 reanalysis</strong></div>
        </>}
      </div>
    </div>
  );
}

// ── Detail panel (bottom slide-up) — Chinese turbine OR German turbine/solar ──
function DetailPanel({ item, monthly, onClose }) {
  if (!item) return null;

  const isGerman = item._type === "de_wind" || item._type === "de_solar";
  const isSolar  = item._type === "de_solar";
  const color    = isSolar ? "#fbbf24" : (CLUSTER_COLORS[item.cluster_id] ?? "#3b82f6");
  const accent   = isSolar ? "#fbbf24" : (item._type === "de_wind" ? "#60a5fa" : "#38bdf8");

  const titleIcon  = isSolar ? "☀️" : "💨";
  const titleLabel = isGerman
    ? `${item.operator} · ${item.capacity_mw} MW · ${item.state}`
    : `Turbine #${item.id} — ${item.cluster_label}`;

  return (
    <div className="turb-panel">
      <div className="tp-header">
        <span className="tp-dot" style={{ background: color }} />
        <span className="tp-title">{titleIcon} {titleLabel}</span>
        <div className="tp-stats">
          {!isGerman && <>
            <span>Perf {item.performance_score}%</span>
            <span>Avail {(item.availability * 100).toFixed(0)}%</span>
            <span>Yaw {item.mean_yaw_misalignment}°</span>
          </>}
          {isGerman && <>
            <span>Built {item.year_built}</span>
            <span>Risk {item.maintenance_risk}</span>
            <span>AEP {(item.est_annual_mwh / 1000).toFixed(0)} GWh/yr</span>
          </>}
        </div>
        <button className="tp-close" onClick={onClose}>✕</button>
      </div>
      <div className="tp-body">
        <div className="tp-section-title">
          {isGerman
            ? `Monthly production estimate (ERA5 reanalysis) · 12 months + 3-month seasonal forecast`
            : `Monthly performance Jan–Sep 2020 (SDWPF SCADA) + 3-month seasonal forecast`}
        </div>
        <MonthlyChart data={monthly} accent={accent} mode={isGerman ? "era5" : "scada"} />
      </div>
    </div>
  );
}

// ── Map controller — programmatic pan/zoom from outside MapContainer ──────────
function MapController({ target }) {
  const map = useMap();
  const prevRef = useRef(null);
  useEffect(() => {
    if (!target || target === prevRef.current) return;
    prevRef.current = target;
    map.flyTo([target.lat, target.lon], target.zoom ?? 13, { duration: 1.2 });
  }, [target, map]);
  return null;
}

// ── Drag-to-select ────────────────────────────────────────────────────────────
function DragSelector({ active, onSelect, onCancel }) {
  const startRef  = useRef(null);
  const activeRef = useRef(active);
  const [rect, setRect] = useState(null);
  const map = useMap();

  useEffect(() => { activeRef.current = active; }, [active]);
  useEffect(() => {
    if (!active && startRef.current) { map.dragging.enable(); startRef.current = null; setRect(null); }
  }, [active, map]);

  useMapEvents({
    mousedown(e) {
      if (!activeRef.current) return;
      map.dragging.disable(); startRef.current = e.latlng;
      setRect({ start: e.latlng, end: e.latlng });
    },
    mousemove(e) { if (!startRef.current) return; setRect(p => p ? { start: p.start, end: e.latlng } : null); },
    mouseup(e) {
      if (!startRef.current) return;
      map.dragging.enable();
      const start = startRef.current; startRef.current = null; setRect(null);
      const bbox = {
        lat_min: Math.min(start.lat, e.latlng.lat), lat_max: Math.max(start.lat, e.latlng.lat),
        lon_min: Math.min(start.lng, e.latlng.lng), lon_max: Math.max(start.lng, e.latlng.lng),
      };
      if (bbox.lat_max - bbox.lat_min > 0.002 && bbox.lon_max - bbox.lon_min > 0.002) onSelect(bbox);
      else onCancel?.();
    },
  });

  if (!rect) return null;
  return <Rectangle bounds={[[rect.start.lat, rect.start.lng], [rect.end.lat, rect.end.lng]]}
    pathOptions={{ color: "#f59e0b", weight: 2, fillOpacity: 0.1, dashArray: "6,4" }} />;
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [mode, setMode] = useState("wind");

  const [turbines,         setTurbines]         = useState([]);
  const [predictions,      setPredictions]      = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [siting,           setSiting]           = useState([]);
  const [solarSiting,      setSolarSiting]      = useState([]);
  const [solarResult,      setSolarResult]      = useState(null);
  const [regionStats,      setRegionStats]      = useState(null);
  const [bbox,             setBbox]             = useState(null);
  const [selecting,        setSelecting]        = useState(false);
  const [loading,          setLoading]          = useState("");
  const [nNew,             setNNew]             = useState(3);
  const [siteMw,           setSiteMw]           = useState(10);
  const [weather,          setWeather]          = useState({ wind_speed_ms: 8, wind_direction_deg: 45, temperature_c: 15 });
  const [solarWeather,     setSolarWeather]     = useState({ ghi_wm2: 400, temperature_c: 25, humidity_pct: 50 });
  const [forecast,         setForecast]         = useState([]);
  const [forecastHour,     setForecastHour]     = useState(0);
  const [forecastBusy,     setForecastBusy]     = useState(false);

  // Map layers
  const [showRadar, setShowRadar] = useState(false);
  const [showCloud, setShowCloud] = useState(false);
  const [radarPaths, setRadarPaths]     = useState([]);
  const [radarFrameIdx, setRadarFrameIdx] = useState(0);
  const cloudUrl = modisUrl();

  // Germany
  const [germanyTurbines,    setGermanyTurbines]    = useState([]);
  const [germanySolar,       setGermanySolar]       = useState([]);
  const [showGermany,        setShowGermany]        = useState(false);
  const [germanyPredictions, setGermanyPredictions] = useState({});
  const [germanyResult,      setGermanyResult]      = useState(null);

  // Business
  const [elecPrice, setElecPrice] = useState(null);

  // Maintenance
  const [maintenanceRisk, setMaintenanceRisk] = useState(null);
  const [maintWindows,    setMaintWindows]    = useState([]);
  const [showMaint,       setShowMaint]       = useState(false);

  // Siting history popups
  const [sitingHistory, setSitingHistory] = useState({});

  // Detail panel (Chinese + German turbines + solar)
  const [selectedItem,   setSelectedItem]   = useState(null);  // {id, _type, ...}
  const [itemMonthly,    setItemMonthly]    = useState(null);  // monthly chart data

  // Map focus & highlight
  const [mapTarget,         setMapTarget]         = useState(null);
  const [highlightedId,     setHighlightedId]     = useState(null);   // Chinese turbine
  const [highlightedSiteRank, setHighlightedSiteRank] = useState(null); // wind siting
  const [highlightedSolarRank, setHighlightedSolarRank] = useState(null); // solar siting

  // Germany operator filter
  const [operatorSearch,    setOperatorSearch]    = useState("");
  const [selectedOperators, setSelectedOperators] = useState(new Set());
  const [showOpFilter,      setShowOpFilter]      = useState(false);

  // ── Load on mount ───────────────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${API}/api/turbines`).then(r => r.json()).then(d => setTurbines(d.turbines || [])).catch(console.error);
  }, []);
  useEffect(() => {
    fetch(`${API}/api/electricity/price`).then(r => r.json()).then(setElecPrice)
      .catch(() => setElecPrice({ price_eur_mwh: 68, live: false }));
  }, []);
  useEffect(() => {
    fetch(`${API}/api/maintenance/risk`).then(r => r.json()).then(setMaintenanceRisk).catch(console.error);
  }, []);

  // Fetch all RainViewer radar frames for animation
  useEffect(() => {
    fetch("https://api.rainviewer.com/public/weather-maps.json")
      .then(r => r.json())
      .then(data => {
        const host = data.host || "https://tilecache.rainviewer.com";
        const past = data.radar?.past || [];
        if (past.length) {
          const paths = past.map(p => host + p.path);
          setRadarPaths(paths);
          setRadarFrameIdx(paths.length - 1);
        }
      }).catch(() => {});
  }, []);

  // Animate radar frames at 600 ms/frame
  useEffect(() => {
    if (!showRadar || radarPaths.length < 2) return;
    const timer = setInterval(() => setRadarFrameIdx(i => (i + 1) % radarPaths.length), 600);
    return () => clearInterval(timer);
  }, [showRadar, radarPaths]);

  // Fly to Germany and load data when Germany toggle is turned on
  useEffect(() => {
    if (!showGermany) return;
    // Fly map to Germany
    setMapTarget({ lat: 51.5, lon: 10.5, zoom: 6 });
    // Load wind turbines if in wind/both mode
    if ((mode === "wind" || mode === "both") && germanyTurbines.length === 0)
      fetch(`${API}/api/germany/turbines`).then(r => r.json()).then(d => setGermanyTurbines(d.turbines || [])).catch(console.error);
    // Load solar parks if in solar/both mode
    if ((mode === "solar" || mode === "both") && germanySolar.length === 0)
      fetch(`${API}/api/germany/solar`).then(r => r.json()).then(d => setGermanySolar(d.parks || [])).catch(console.error);
  }, [showGermany]);

  // When mode changes while Germany is on, load whichever dataset is missing
  useEffect(() => {
    if (!showGermany) return;
    if ((mode === "solar" || mode === "both") && germanySolar.length === 0)
      fetch(`${API}/api/germany/solar`).then(r => r.json()).then(d => setGermanySolar(d.parks || [])).catch(console.error);
    if ((mode === "wind" || mode === "both") && germanyTurbines.length === 0)
      fetch(`${API}/api/germany/turbines`).then(r => r.json()).then(d => setGermanyTurbines(d.turbines || [])).catch(console.error);
  }, [mode]);

  // Sync forecast → weather sliders
  useEffect(() => {
    if (!forecast.length) return;
    const h = forecast[Math.min(forecastHour, forecast.length - 1)];
    if (!h) return;
    setWeather({ wind_speed_ms: Math.round((h.wind_speed_ms ?? 8) * 10) / 10, wind_direction_deg: Math.round(h.wind_direction_deg ?? 45), temperature_c: Math.round(h.temperature_c ?? 15) });
    if (h.shortwave_radiation_wm2 != null)
      setSolarWeather(sw => ({ ...sw, ghi_wm2: Math.round(h.shortwave_radiation_wm2), temperature_c: Math.round(h.temperature_c ?? 25) }));
  }, [forecastHour, forecast]);

  // ── Actions ─────────────────────────────────────────────────────────────────
  async function fetchLiveForecast() {
    setForecastBusy(true);
    try { const r = await fetch(`${API}/api/weather?lat=40.5&lon=108.5&hours=48`).then(r => r.json()); setForecast(r.forecast || []); setForecastHour(0); } catch {}
    setForecastBusy(false);
  }
  async function predict() {
    setLoading("Predicting wind power output…");
    try {
      const res = await fetch(`${API}/api/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ weather }) }).then(r => r.json());
      const map = {}; (res.turbines || []).forEach(t => { map[t.TurbID] = t.predicted_power_kw; });
      setPredictions(map); setPredictionResult({ total_mw: res.total_predicted_mw, n_turbines: res.n_turbines });
    } catch (e) { console.error(e); }
    setLoading("");
  }
  async function predictSolar() {
    setLoading("Predicting solar output…");
    try {
      const now = new Date();
      const res = await fetch(`${API}/api/solar/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ghi_wm2: solarWeather.ghi_wm2, temperature_c: solarWeather.temperature_c, humidity_pct: solarWeather.humidity_pct, hour: now.getHours() + now.getMinutes() / 60, month: now.getMonth() + 1, day_of_year: Math.ceil((now - new Date(now.getFullYear(), 0, 0)) / 86400000) }) }).then(r => r.json());
      setSolarResult(res);
    } catch (e) { console.error(e); }
    setLoading("");
  }
  async function predictGermany() {
    if (!bbox) return alert("Draw a region first.");
    setLoading("Predicting German turbines…");
    try {
      const res = await fetch(`${API}/api/germany/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ weather, bbox }) }).then(r => r.json());
      const map = {}; (res.turbines || []).forEach(t => { map[t.id] = t.predicted_power_kw; });
      setGermanyPredictions(map); setGermanyResult({ total_mw: res.total_predicted_mw, n_turbines: res.n_turbines });
    } catch (e) { console.error(e); }
    setLoading("");
  }
  async function runSiting() {
    if (!bbox) return alert("Draw a region first.");
    setLoading("Finding best wind turbine sites…");
    try { const res = await fetch(`${API}/api/siting`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...bbox, n_turbines: nNew }) }).then(r => r.json()); setSiting(res.locations || []); } catch (e) { console.error(e); }
    setLoading("");
  }
  async function runSolarSiting() {
    if (!bbox) return alert("Draw a region first.");
    setLoading("Finding best solar sites…");
    try { const res = await fetch(`${API}/api/solar/siting`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...bbox, n_sites: nNew, site_capacity_mw: siteMw }) }).then(r => r.json()); setSolarSiting(res.locations || []); } catch (e) { console.error(e); }
    setLoading("");
  }
  async function loadRegion() {
    if (!bbox) return;
    setLoading("Loading region stats…");
    try { const res = await fetch(`${API}/api/region/summary`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(bbox) }).then(r => r.json()); setRegionStats(res); } catch (e) { console.error(e); }
    setLoading("");
  }
  async function fetchMaintWindows() {
    setLoading("Finding low-wind maintenance windows…");
    try { const res = await fetch(`${API}/api/maintenance/windows?lat=40.5&lon=108.5&days=16`).then(r => r.json()); setMaintWindows(res.windows || []); } catch (e) { console.error(e); }
    setLoading("");
  }
  async function fetchSitingHistory(lat, lon) {
    const key = `${lat.toFixed(3)},${lon.toFixed(3)}`;
    if (sitingHistory[key]) return;
    try { const res = await fetch(`${API}/api/historical/stats?lat=${lat}&lon=${lon}`).then(r => r.json()); setSitingHistory(h => ({ ...h, [key]: res })); } catch {}
  }
  async function openTurbine(t) {
    setSelectedItem({ ...t, _type: "cn_wind" });
    setItemMonthly(null);
    setHighlightedId(t.id);
    try {
      const res = await fetch(`${API}/api/turbines/${t.id}/monthly`).then(r => r.json());
      setItemMonthly(res);
    } catch {}
  }

  async function openGermanSite(item, type) {
    setSelectedItem({ ...item, _type: type });
    setItemMonthly(null);
    const cap = item.capacity_mw ?? 2.0;
    const siteType = type === "de_solar" ? "solar" : "wind";
    try {
      const res = await fetch(`${API}/api/site/performance?lat=${item.lat}&lon=${item.lon}&capacity_mw=${cap}&site_type=${siteType}`).then(r => r.json());
      setItemMonthly(res);
    } catch {}
  }
  function focusTurbine(turbineId) {
    const t = turbines.find(t => t.id === turbineId);
    if (!t) return;
    setMapTarget({ lat: t.lat, lon: t.lon, zoom: 15 });
    setHighlightedId(turbineId);
    // If a region is selected and turbine is outside it, clear the region
    if (bbox) {
      const inside = t.lat >= bbox.lat_min && t.lat <= bbox.lat_max &&
                     t.lon >= bbox.lon_min && t.lon <= bbox.lon_max;
      if (!inside) setBbox(null);
    }
  }

  function clearAll() {
    setBbox(null); setSiting([]); setSolarSiting([]); setRegionStats(null);
    setPredictions({}); setPredictionResult(null); setSolarResult(null);
    setGermanyPredictions({}); setGermanyResult(null); setSelecting(false);
    setHighlightedSiteRank(null); setHighlightedSolarRank(null);
  }

  // ── Derived ─────────────────────────────────────────────────────────────────
  const showWind  = mode === "wind"  || mode === "both";
  const showSolar = mode === "solar" || mode === "both";

  const displayTurbines = useMemo(() =>
    bbox ? turbines.filter(t => t.lat >= bbox.lat_min && t.lat <= bbox.lat_max && t.lon >= bbox.lon_min && t.lon <= bbox.lon_max) : turbines,
    [turbines, bbox]);

  const germanOperators = useMemo(() =>
    [...new Set(germanyTurbines.map(t => t.operator))].sort(),
    [germanyTurbines]);

  const displayGermany = useMemo(() => {
    let list = germanyTurbines;
    if (bbox) list = list.filter(t => t.lat >= bbox.lat_min && t.lat <= bbox.lat_max && t.lon >= bbox.lon_min && t.lon <= bbox.lon_max);
    if (selectedOperators.size > 0) list = list.filter(t => selectedOperators.has(t.operator));
    return list;
  }, [germanyTurbines, bbox, selectedOperators]);
  const curHour = forecast[Math.min(forecastHour, forecast.length - 1)];

  const farmCapMw = predictionResult ? predictionResult.n_turbines * 1.5 : 192;
  const farmCF    = predictionResult && farmCapMw > 0 ? predictionResult.total_mw / farmCapMw : 0.32;
  const lcoe      = elecPrice && predictionResult ? calcLCOE(farmCapMw, farmCF, elecPrice.price_eur_mwh) : null;
  const earningNow = predictionResult && elecPrice ? Math.round(predictionResult.total_mw * elecPrice.price_eur_mwh) : null;

  const deCapMw  = germanyResult ? germanyResult.n_turbines * 3.5 : 0;
  const deCF     = germanyResult && deCapMw > 0 ? germanyResult.total_mw / deCapMw : 0.28;
  const deLcoe   = elecPrice && germanyResult ? calcLCOE(deCapMw, deCF, elecPrice.price_eur_mwh) : null;
  const deEarning = germanyResult && elecPrice ? Math.round(germanyResult.total_mw * elecPrice.price_eur_mwh) : null;

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      {/* ── Sidebar ────────────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="logo-row">
          <div className="logo">⚡ Wind &amp; Solar AI</div>
          {elecPrice && (
            <div className={`price-badge ${elecPrice.live ? "live" : "est"}`}>
              €{elecPrice.price_eur_mwh}/MWh {elecPrice.live ? "●" : "~"}
            </div>
          )}
        </div>
        <p className="subtitle">
          {turbines.length} Chinese turbines
          {germanyTurbines.length > 0 && ` · ${germanyTurbines.length} German wind`}
          {germanySolar.length > 0 && ` · ${germanySolar.length} DE solar`}
        </p>

        <div className="mode-toggle">
          <button className={`mode-btn ${mode === "wind"  ? "active" : ""}`} onClick={() => setMode("wind")}>💨 Wind</button>
          <button className={`mode-btn ${mode === "solar" ? "active" : ""}`} onClick={() => setMode("solar")}>☀️ Solar</button>
          <button className={`mode-btn ${mode === "both"  ? "active" : ""}`} onClick={() => setMode("both")}>⚡+☀️ Both</button>
        </div>

        {showWind && (
          <section>
            <h3>Wind Cluster Legend</h3>
            {Object.entries(CLUSTER_LABELS).map(([id, label]) => (
              <div key={id} className="legend-item"><span className="dot" style={{ background: CLUSTER_COLORS[id] }} /><span>{label}</span></div>
            ))}
          </section>
        )}
        {showSolar && (
          <section>
            <h3>Solar Legend</h3>
            <div className="legend-item"><span className="dot" style={{ background: "#f97316", borderRadius: "3px" }} /><span>Siting Candidate</span></div>
            <div className="legend-item"><span className="dot" style={{ background: "#fde68a", borderRadius: "3px", border: "1px solid #fbbf24" }} /><span>🇩🇪 German Solar Park (≥1 MW)</span></div>
          </section>
        )}

        <section>
          <h3>Live Weather</h3>
          <div className="weather-toolbar">
            <button className={`btn small ${forecastBusy ? "selecting" : ""}`} onClick={fetchLiveForecast}>{forecastBusy ? "Loading…" : "🌤 Forecast"}</button>
            <button className={`btn small ${showCloud  ? "active-toggle" : ""}`} onClick={() => setShowCloud(v => !v)}>☁ Clouds</button>
            <button className={`btn small ${showRadar  ? "active-toggle" : ""}`} onClick={() => setShowRadar(v => !v)}
              title={radarPaths.length > 1 ? `Animated — ${radarPaths.length} frames` : "Radar"}>
              📡 Radar{radarPaths.length > 1 ? " ▶" : ""}
            </button>
            <button className={`btn small ${showGermany ? "active-toggle" : ""}`} onClick={() => setShowGermany(v => !v)}>🇩🇪 Germany</button>
          </div>
          {forecast.length > 0 && <>
            <label>Forecast: +{forecastHour}h
              <input type="range" min="0" max={forecast.length - 1} step="1" value={forecastHour} onChange={e => setForecastHour(+e.target.value)} />
            </label>
            {curHour && (
              <div className="forecast-pill">
                <span>🌬 {(curHour.wind_speed_ms ?? 0).toFixed(1)} m/s</span>
                <span>📐 {Math.round(curHour.wind_direction_deg ?? 0)}°</span>
                <span>🌡 {Math.round(curHour.temperature_c ?? 0)}°C</span>
                {curHour.cloud_cover_pct != null && <span>☁ {curHour.cloud_cover_pct}%</span>}
                {curHour.shortwave_radiation_wm2 != null && <span>☀ {Math.round(curHour.shortwave_radiation_wm2)} W/m²</span>}
              </div>
            )}
          </>}
        </section>

        {showWind && (
          <section>
            <h3>Wind Scenario</h3>
            <label>Wind Speed (m/s)
              <input type="range" min="0" max="25" step="0.5" value={weather.wind_speed_ms} onChange={e => setWeather(w => ({ ...w, wind_speed_ms: +e.target.value }))} />
              <span className="val">{weather.wind_speed_ms} m/s</span>
            </label>
            <label>Wind Direction (°)
              <input type="range" min="0" max="360" step="5" value={weather.wind_direction_deg} onChange={e => setWeather(w => ({ ...w, wind_direction_deg: +e.target.value }))} />
              <span className="val">{weather.wind_direction_deg}°</span>
            </label>
            <label>Temperature (°C)
              <input type="range" min="-10" max="40" step="1" value={weather.temperature_c} onChange={e => setWeather(w => ({ ...w, temperature_c: +e.target.value }))} />
              <span className="val">{weather.temperature_c}°C</span>
            </label>
            <button className="btn primary" onClick={predict}>⚡ Predict Wind Output</button>
            {showGermany && showWind && (
              <div style={{ marginTop: 6 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                  <button className={`btn small ${showOpFilter ? "active-toggle" : ""}`} onClick={() => setShowOpFilter(v => !v)}>
                    🏢 Filter by operator {selectedOperators.size > 0 ? `(${selectedOperators.size})` : ""}
                  </button>
                  {selectedOperators.size > 0 && (
                    <button className="btn small danger" style={{ padding: "3px 6px", fontSize: 10 }} onClick={() => setSelectedOperators(new Set())}>✕ Clear</button>
                  )}
                </div>
                {showOpFilter && (
                  <div className="op-filter">
                    <input
                      className="op-search"
                      placeholder="Search operator…"
                      value={operatorSearch}
                      onChange={e => setOperatorSearch(e.target.value)}
                    />
                    <div className="op-list">
                      {germanOperators.filter(op => op.toLowerCase().includes(operatorSearch.toLowerCase())).map(op => (
                        <label key={op} className={`op-item ${selectedOperators.has(op) ? "op-selected" : ""}`}>
                          <input type="checkbox" checked={selectedOperators.has(op)}
                            onChange={() => setSelectedOperators(prev => {
                              const next = new Set(prev);
                              next.has(op) ? next.delete(op) : next.add(op);
                              return next;
                            })} />
                          <span className="op-name">{op}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                )}
                {bbox && displayGermany.length > 0 && (
                  <button className="btn de-btn" style={{ marginTop: 4, width: "100%" }} onClick={predictGermany}>
                    🇩🇪 Predict {selectedOperators.size > 0 ? selectedOperators.size + " operator(s)" : "Germany"} ({displayGermany.length}T)
                  </button>
                )}
              </div>
            )}
          </section>
        )}

        {showWind && predictionResult && (
          <section className="stats-box">
            <h3>Chinese Farm Output</h3>
            <div className="stat"><span>Total Farm</span><strong>{predictionResult.total_mw} MW</strong></div>
            <div className="stat"><span>Turbines</span><strong>{predictionResult.n_turbines}</strong></div>
            <div className="stat"><span>Avg/turbine</span><strong>{predictionResult.n_turbines > 0 ? ((predictionResult.total_mw * 1000) / predictionResult.n_turbines).toFixed(0) : "—"} kW</strong></div>
            {bbox && Object.keys(predictions).length > 0 && (
              <div className="stat"><span>Selected region</span><strong>{(displayTurbines.reduce((s, t) => s + (predictions[t.id] || 0), 0) / 1000).toFixed(2)} MW ({displayTurbines.length}T)</strong></div>
            )}
            {earningNow != null && <div className="stat earning"><span>💰 Earning right now</span><strong>€{earningNow.toLocaleString()}/hr</strong></div>}
            {lcoe && <>
              <div className="lcoe-divider">LCOE &amp; Business Metrics</div>
              <div className="stat"><span>LCOE</span><strong>€{lcoe.lcoe}/MWh</strong></div>
              <div className="stat"><span>Payback</span><strong>{lcoe.payback} years</strong></div>
              <div className="stat"><span>Est. IRR</span><strong>{lcoe.irr}%</strong></div>
              <div className="stat"><span>Annual Revenue</span><strong>€{(lcoe.annRevEur / 1e6).toFixed(1)}M</strong></div>
              <div className="stat"><span>CO₂ Offset</span><strong>{(lcoe.co2TonYr / 1000).toFixed(1)}k t/yr</strong></div>
            </>}
          </section>
        )}

        {germanyResult && (
          <section className="stats-box de-box">
            <h3>🇩🇪 Germany Region</h3>
            <div className="stat"><span>Total Power</span><strong>{germanyResult.total_mw} MW</strong></div>
            <div className="stat"><span>Turbines</span><strong>{germanyResult.n_turbines}</strong></div>
            {deEarning != null && <div className="stat earning"><span>💰 Earning right now</span><strong>€{deEarning.toLocaleString()}/hr</strong></div>}
            {deLcoe && <>
              <div className="lcoe-divider">Business</div>
              <div className="stat"><span>LCOE</span><strong>€{deLcoe.lcoe}/MWh</strong></div>
              <div className="stat"><span>Payback</span><strong>{deLcoe.payback} yrs</strong></div>
              <div className="stat"><span>IRR</span><strong>{deLcoe.irr}%</strong></div>
              <div className="stat"><span>CO₂</span><strong>{(deLcoe.co2TonYr / 1000).toFixed(1)}k t/yr</strong></div>
            </>}
          </section>
        )}

        {showSolar && (
          <section>
            <h3>Solar Scenario</h3>
            <label>GHI (W/m²)
              <input type="range" min="0" max="1000" step="10" value={solarWeather.ghi_wm2} onChange={e => setSolarWeather(s => ({ ...s, ghi_wm2: +e.target.value }))} />
              <span className="val">{solarWeather.ghi_wm2} W/m²</span>
            </label>
            <label>Temperature (°C)
              <input type="range" min="-10" max="50" step="1" value={solarWeather.temperature_c} onChange={e => setSolarWeather(s => ({ ...s, temperature_c: +e.target.value }))} />
              <span className="val">{solarWeather.temperature_c}°C</span>
            </label>
            <label>Humidity (%)
              <input type="range" min="0" max="100" step="1" value={solarWeather.humidity_pct} onChange={e => setSolarWeather(s => ({ ...s, humidity_pct: +e.target.value }))} />
              <span className="val">{solarWeather.humidity_pct}%</span>
            </label>
            <button className="btn solar-btn" onClick={predictSolar}>☀️ Predict Solar Output</button>
          </section>
        )}
        {showSolar && solarResult && (
          <section className="stats-box">
            <h3>Solar Prediction</h3>
            <div className="stat"><span>Capacity Factor</span><strong>{(solarResult.capacity_factor * 100).toFixed(1)}%</strong></div>
            <div className="stat"><span>Per 1 MW</span><strong>{solarResult.estimated_power_per_mw_kw} kW</strong></div>
            <div className="stat"><span>Per 10 MW park</span><strong>{(solarResult.estimated_power_per_mw_kw * 10 / 1000).toFixed(2)} MW</strong></div>
            <div className="stat"><span>Model</span><strong>{solarResult.model}</strong></div>
          </section>
        )}

        <section>
          <h3>Region Analysis</h3>
          <button className={`btn ${selecting ? "selecting" : ""}`} onClick={() => { if (!selecting) clearAll(); setSelecting(s => !s); }}>
            {selecting ? "✏ Drag to draw…" : "🗺 Select Region"}
          </button>
          {bbox && !selecting && (
            <div className="region-actions">
              {showWind && <button className="btn" onClick={loadRegion}>📊 Wind Stats</button>}
              <div className="siting-row"><span>Find</span><input type="number" value={nNew} min="1" max="10" onChange={e => setNNew(+e.target.value)} /><span>sites</span></div>
              {showWind  && <button className="btn primary" onClick={runSiting}>📍 Best Wind Sites</button>}
              {showSolar && <><div className="siting-row"><input type="number" value={siteMw} min="1" max="500" onChange={e => setSiteMw(+e.target.value)} /><span>MW each</span></div>
                <button className="btn solar-btn" onClick={runSolarSiting}>🌞 Best Solar Sites</button></>}
              <button className="btn danger" onClick={clearAll}>✕ Clear</button>
            </div>
          )}
        </section>

        {regionStats && regionStats.n_turbines > 0 && (
          <section className="stats-box">
            <h3>Region Stats ({regionStats.n_turbines} turbines)</h3>
            <div className="stat"><span>Avg Performance</span><strong>{regionStats.avg_performance_score}%</strong></div>
            <div className="stat"><span>Hist. Avg Power</span><strong>{(regionStats.total_avg_power_kw / 1000).toFixed(1)} MW</strong></div>
            {Object.keys(predictions).length > 0 && <div className="stat"><span>⚡ Predicted</span><strong>{(displayTurbines.reduce((s, t) => s + (predictions[t.id] || 0), 0) / 1000).toFixed(2)} MW</strong></div>}
            <div className="stat"><span>Availability</span><strong>{(regionStats.avg_availability * 100).toFixed(1)}%</strong></div>
            <div className="stat"><span>Avg Yaw Error</span><strong>{regionStats.avg_yaw_misalignment_deg}°</strong></div>
          </section>
        )}

        <section>
          <h3>
            🔧 Maintenance
            {maintenanceRisk?.critical_count > 0 && <span className="risk-badge critical"> {maintenanceRisk.critical_count} critical</span>}
            {maintenanceRisk?.high_count     > 0 && <span className="risk-badge high"> {maintenanceRisk.high_count} high</span>}
          </h3>
          <button className={`btn small ${showMaint ? "active-toggle" : ""}`} onClick={() => setShowMaint(v => !v)}>
            {showMaint ? "▲ Hide" : "▼ Show"} Risk Analysis
          </button>
          {showMaint && maintenanceRisk && (
            <>
              {maintenanceRisk.top_10_at_risk.slice(0, 5).map(t => (
                <div key={t.turbine_id}
                  className={`maint-row maint-clickable ${highlightedId === t.turbine_id ? "maint-active" : ""}`}
                  onClick={() => focusTurbine(t.turbine_id)}
                  title="Click to locate on map">
                  <span className={`risk-dot ${t.risk_level.toLowerCase()}`} />
                  <div style={{ flex: 1 }}>
                    <div className="maint-title">T#{t.turbine_id} — <span className={`risk-text ${t.risk_level.toLowerCase()}`}>{t.risk_level}</span></div>
                    <div className="maint-sub">{t.urgency}</div>
                    <div className="maint-issue">{t.issues[0]}</div>
                  </div>
                  <span className="maint-pin">📍</span>
                </div>
              ))}
              <button className="btn small" style={{ marginTop: 8 }} onClick={fetchMaintWindows}>🗓 Find Low-Wind Windows (16d)</button>
              {maintWindows.length > 0 && (
                <div style={{ marginTop: 6 }}>
                  <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>Best windows to take turbines offline:</div>
                  {maintWindows.map((w, i) => (
                    <div key={i} className="maint-window">
                      <span className={`window-badge ${w.rating.toLowerCase()}`}>{w.rating}</span>
                      <div>
                        <div className="maint-title">{w.start.slice(5, 16)}</div>
                        <div className="maint-sub">{w.duration_hours}h · avg {w.avg_wind_ms} m/s</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </section>

        {showWind && siting.length > 0 && (
          <section className="stats-box">
            <h3>Best Wind Sites</h3>
            <p className="siting-note">Hub-height wind × elevation × slope × wake · 500 m min. physical spacing (fixed, not zoom-dependent)</p>
            {siting.map(s => (
              <div key={s.rank}
                className={`siting-result siting-clickable ${highlightedSiteRank === s.rank ? "siting-active" : ""}`}
                onClick={() => { setMapTarget({ lat: s.lat, lon: s.lon, zoom: 14 }); setHighlightedSiteRank(s.rank); }}
                title="Click to show on map">
                <span className="rank">#{s.rank} 📍</span>
                <div>
                  <div>{s.estimated_avg_power_kw} kW · {s.elevation_m} m</div>
                  <div className="small">{s.wind_speed_100m_ms} m/s @100m · {s.wake_exposure_pct}% wake · score {s.siting_score}</div>
                </div>
              </div>
            ))}
            <div className="stat total"><span>Total AEP</span><strong>{siting.reduce((a, s) => a + (s.estimated_aep_mwh_per_year || 0), 0).toLocaleString()} MWh/yr</strong></div>
            {elecPrice && <div className="stat total"><span>Est. Revenue</span><strong>€{Math.round(siting.reduce((a, s) => a + (s.estimated_aep_mwh_per_year || 0), 0) * elecPrice.price_eur_mwh).toLocaleString()}</strong></div>}
          </section>
        )}
        {showSolar && solarSiting.length > 0 && (
          <section className="stats-box">
            <h3>Best Solar Sites</h3>
            <p className="siting-note">GHI × elevation × slope × south aspect · 300 m spacing</p>
            {solarSiting.map(s => (
              <div key={s.rank}
                className={`siting-result siting-clickable ${highlightedSolarRank === s.rank ? "siting-active" : ""}`}
                onClick={() => { setMapTarget({ lat: s.lat, lon: s.lon, zoom: 14 }); setHighlightedSolarRank(s.rank); }}
                title="Click to show on map">
                <span className="rank solar-rank">#{s.rank} 📍</span>
                <div>
                  <div>{s.site_capacity_mw} MW · {s.estimated_avg_power_kw} kW avg · CF {(s.estimated_capacity_factor * 100).toFixed(1)}%</div>
                  <div className="small">{s.avg_ghi_wm2} W/m² GHI · {s.elevation_m} m · score {s.siting_score}</div>
                </div>
              </div>
            ))}
            <div className="stat total"><span>Total AEP</span><strong>{solarSiting.reduce((a, s) => a + (s.estimated_aep_mwh_per_year || 0), 0).toLocaleString()} MWh/yr</strong></div>
          </section>
        )}

        {loading && <div className="loading-bar">{loading}</div>}
      </aside>

      {/* ── Map ──────────────────────────────────────────────────────────── */}
      <main className={`map-wrapper ${selecting ? "crosshair" : ""}`}>
        <MapContainer center={[40.5, 108.5]} zoom={12} style={{ height: "100%", width: "100%" }}>
          <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" attribution="Tiles © Esri" />
          {showCloud && <TileLayer url={cloudUrl} opacity={0.65} attribution="NASA GIBS" />}

          {/* Flicker-free animated radar — native Leaflet layers */}
          <RadarLayer paths={radarPaths} frameIdx={radarFrameIdx} show={showRadar} opacity={0.6} />

          <MapController target={mapTarget} />
          <DragSelector active={selecting} onSelect={b => { setBbox(b); setSelecting(false); }} onCancel={() => setSelecting(false)} />
          {bbox && !selecting && <Rectangle bounds={[[bbox.lat_min, bbox.lon_min], [bbox.lat_max, bbox.lon_max]]} pathOptions={{ color: "#f59e0b", weight: 2, dashArray: "6,4", fillOpacity: 0.05 }} />}

          {/* Chinese turbines */}
          {showWind && displayTurbines.map(t => {
            const predKw = predictions[t.id];
            const color  = CLUSTER_COLORS[t.cluster_id] ?? "#6b7280";
            const r      = predKw ? Math.max(5, Math.min(14, predKw / 100)) : 6;
            const isSelected    = selectedItem?.id === t.id && selectedItem?._type === "cn_wind";
            const isHighlighted = highlightedId === t.id;
            return (
              <CircleMarker key={t.id} center={[t.lat, t.lon]}
                radius={isSelected || isHighlighted ? r + 4 : r}
                color={isHighlighted ? "#f59e0b" : "#fff"}
                weight={isSelected || isHighlighted ? 3 : 1}
                fillColor={color} fillOpacity={0.95}
                eventHandlers={{ click: e => { e.target.openPopup(); openTurbine(t); setHighlightedId(t.id); } }}>
                <Popup>
                  <div className="popup">
                    <strong>Turbine #{t.id}</strong>
                    <table><tbody>
                      <tr><td>Cluster</td><td>{t.cluster_label}</td></tr>
                      <tr><td>Performance</td><td>{t.performance_score}%</td></tr>
                      <tr><td>Availability</td><td>{(t.availability * 100).toFixed(1)}%</td></tr>
                      <tr><td>Avg Power</td><td>{t.mean_patv_kw} kW</td></tr>
                      <tr><td>Yaw Error</td><td>{t.mean_yaw_misalignment}°</td></tr>
                      {predKw && <tr><td><strong>Predicted</strong></td><td><strong>{predKw.toFixed(0)} kW</strong></td></tr>}
                    </tbody></table>
                    <div style={{ marginTop: 6, fontSize: 11, color: "#60a5fa", cursor: "pointer" }} onClick={() => openTurbine(t)}>
                      📈 Monthly performance + forecast
                    </div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}

          {/* German turbines */}
          {showGermany && showWind && displayGermany.map(t => {
            const predKw    = germanyPredictions[t.id];
            const riskColor = t.maintenance_risk === "HIGH" ? "#ef4444" : t.maintenance_risk === "MEDIUM" ? "#f59e0b" : "#22c55e";
            const isActive  = selectedItem?.id === t.id && selectedItem?._type === "de_wind";
            return (
              <CircleMarker key={`de-${t.id}`} center={[t.lat, t.lon]}
                radius={predKw ? Math.max(3, Math.min(10, predKw / 1000)) : 4}
                color={isActive ? "#fff" : riskColor} weight={isActive ? 3 : 1}
                fillColor="#60a5fa" fillOpacity={isActive ? 1.0 : 0.65}
                eventHandlers={{ click: e => { e.target.openPopup(); openGermanSite(t, "de_wind"); } }}>
                <Popup>
                  <div className="popup">
                    <strong>🇩🇪 {t.operator}</strong>
                    <table><tbody>
                      <tr><td>Capacity</td><td>{t.capacity_mw} MW</td></tr>
                      <tr><td>Built</td><td>{t.year_built} ({t.age_years} yrs)</td></tr>
                      <tr><td>State</td><td>{t.state}</td></tr>
                      <tr><td>Est. AEP</td><td>{t.est_annual_mwh?.toLocaleString()} MWh/yr</td></tr>
                      <tr><td>Est. Revenue</td><td>€{t.est_annual_revenue_eur?.toLocaleString()}/yr</td></tr>
                      <tr><td>Build Cost</td><td>€{(t.est_capex_eur / 1e6).toFixed(1)}M</td></tr>
                      <tr><td>Maint. Risk</td><td style={{ color: riskColor, fontWeight: 600 }}>{t.maintenance_risk}</td></tr>
                      {predKw && <tr><td><strong>Predicted now</strong></td><td><strong>{predKw.toFixed(0)} kW</strong></td></tr>}
                    </tbody></table>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}

          {/* German solar parks */}
          {showGermany && showSolar && germanySolar.map(p => {
            const r = Math.max(3, Math.min(11, p.capacity_mw * 1.5));
            return (
              <CircleMarker key={`de-sol-${p.id}`} center={[p.lat, p.lon]}
                radius={selectedItem?.id === p.id && selectedItem?._type === "de_solar" ? r + 3 : r}
                color={selectedItem?.id === p.id && selectedItem?._type === "de_solar" ? "#fff" : "#fbbf24"}
                weight={selectedItem?.id === p.id && selectedItem?._type === "de_solar" ? 3 : 1.5}
                fillColor="#fde68a" fillOpacity={0.8}
                eventHandlers={{ click: e => { e.target.openPopup(); openGermanSite(p, "de_solar"); } }}>
                <Popup>
                  <div className="popup">
                    <strong>☀️ {p.operator}</strong>
                    <table><tbody>
                      <tr><td>Capacity</td><td>{p.capacity_mw} MW</td></tr>
                      <tr><td>Built</td><td>{p.year_built} ({p.age_years} yrs)</td></tr>
                      <tr><td>State</td><td>{p.state}</td></tr>
                      <tr><td>Est. AEP</td><td>{p.est_annual_mwh?.toLocaleString()} MWh/yr</td></tr>
                      <tr><td>Est. Revenue</td><td>€{p.est_annual_revenue_eur?.toLocaleString()}/yr</td></tr>
                      <tr><td>Build Cost</td><td>€{(p.est_capex_eur / 1e6).toFixed(1)}M</td></tr>
                    </tbody></table>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}

          {/* Wind siting candidates */}
          {showWind && siting.map(s => (
            <CircleMarker key={`ws-${s.rank}`} center={[s.lat, s.lon]}
              radius={12} color="#f97316" weight={2} fillColor="#f97316" fillOpacity={0.75}
              eventHandlers={{ click: e => { e.target.openPopup(); fetchSitingHistory(s.lat, s.lon); } }}>
              <Popup>
                <div className="popup">
                  <strong>💨 Wind Site #{s.rank}</strong>
                  <table><tbody>
                    <tr><td>Score</td><td>{s.siting_score}</td></tr>
                    <tr><td>Elevation</td><td>{s.elevation_m} m</td></tr>
                    <tr><td>Wind @100m</td><td>{s.wind_speed_100m_ms} m/s</td></tr>
                    <tr><td>Wake</td><td>{s.wake_exposure_pct}%</td></tr>
                    <tr><td>Est. Power</td><td>{s.estimated_avg_power_kw} kW</td></tr>
                    <tr><td>AEP</td><td>{(s.estimated_aep_mwh_per_year ?? 0).toLocaleString()} MWh/yr</td></tr>
                    {elecPrice && <tr><td>Annual Revenue</td><td>€{Math.round((s.estimated_aep_mwh_per_year ?? 0) * elecPrice.price_eur_mwh).toLocaleString()}</td></tr>}
                  </tbody></table>
                  {(() => {
                    const key = `${s.lat.toFixed(3)},${s.lon.toFixed(3)}`;
                    const hist = sitingHistory[key];
                    if (!hist) return <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 6 }}>Loading 1-yr history…</div>;
                    return <div style={{ marginTop: 8 }}>
                      <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 4 }}>1-Year ERA5 History</div>
                      <table><tbody>
                        <tr><td>Avg Wind @100m</td><td>{hist.annual.avg_wind_ms} m/s</td></tr>
                        <tr><td>Annual Wind CF</td><td>{(hist.annual.wind_capacity_factor * 100).toFixed(1)}%</td></tr>
                        <tr><td>Best month</td><td>{hist.annual.best_wind_month}</td></tr>
                        <tr><td>Worst month</td><td>{hist.annual.worst_wind_month}</td></tr>
                      </tbody></table>
                    </div>;
                  })()}
                </div>
              </Popup>
            </CircleMarker>
          ))}

          {/* Solar siting candidates */}
          {showSolar && solarSiting.map(s => (
            <CircleMarker key={`ss-${s.rank}`} center={[s.lat, s.lon]}
              radius={14} color="#fbbf24" weight={2} fillColor="#fbbf24" fillOpacity={0.8}
              eventHandlers={{ click: e => { e.target.openPopup(); fetchSitingHistory(s.lat, s.lon); } }}>
              <Popup>
                <div className="popup">
                  <strong>☀️ Solar Site #{s.rank}</strong>
                  <table><tbody>
                    <tr><td>Score</td><td>{s.siting_score}</td></tr>
                    <tr><td>Elevation</td><td>{s.elevation_m} m</td></tr>
                    <tr><td>Avg GHI</td><td>{s.avg_ghi_wm2} W/m²</td></tr>
                    <tr><td>Capacity</td><td>{s.site_capacity_mw} MW</td></tr>
                    <tr><td>AEP</td><td>{(s.estimated_aep_mwh_per_year ?? 0).toLocaleString()} MWh/yr</td></tr>
                    {elecPrice && <tr><td>Annual Revenue</td><td>€{Math.round((s.estimated_aep_mwh_per_year ?? 0) * elecPrice.price_eur_mwh).toLocaleString()}</td></tr>}
                  </tbody></table>
                  {(() => {
                    const key = `${s.lat.toFixed(3)},${s.lon.toFixed(3)}`;
                    const hist = sitingHistory[key];
                    if (!hist) return <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 6 }}>Loading 1-yr history…</div>;
                    return <div style={{ marginTop: 8 }}>
                      <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 4 }}>1-Year ERA5 Solar</div>
                      <table><tbody>
                        <tr><td>Avg GHI</td><td>{hist.annual.avg_ghi_wm2} W/m²</td></tr>
                        <tr><td>Solar CF</td><td>{(hist.annual.solar_capacity_factor * 100).toFixed(1)}%</td></tr>
                        <tr><td>Best month</td><td>{hist.annual.best_solar_month}</td></tr>
                      </tbody></table>
                    </div>;
                  })()}
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>

        {bbox && (
          <div className="bbox-badge">
            {showWind  && `${displayTurbines.length} Chinese turbines`}
            {showGermany && showWind  && displayGermany.length > 0 && ` · ${displayGermany.length} DE wind`}
            {showGermany && showSolar && germanySolar.filter(p => p.lat >= bbox.lat_min && p.lat <= bbox.lat_max && p.lon >= bbox.lon_min && p.lon <= bbox.lon_max).length > 0 &&
              ` · ${germanySolar.filter(p => p.lat >= bbox.lat_min && p.lat <= bbox.lat_max && p.lon >= bbox.lon_min && p.lon <= bbox.lon_max).length} DE solar`}
            {showSolar && solarSiting.length > 0 && ` · ${solarSiting.length} solar sites`}
          </div>
        )}

        {/* Detail panel — Chinese turbine / German turbine / German solar */}
        <DetailPanel
          item={selectedItem}
          monthly={itemMonthly}
          onClose={() => { setSelectedItem(null); setItemMonthly(null); setHighlightedId(null); }}
        />
      </main>
    </div>
  );
}
