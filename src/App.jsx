import { useEffect, useMemo, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const API = "http://localhost:8000";

function useApi(path) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}${path}`)
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json();
      })
      .then(setData)
      .catch((err) => setError(err.message));
  }, [path]);

  return { data, error };
}

const METHOD_LABELS = {
  no_augmentation: "No augmentation",
  bootstrap: "Bootstrap (self-resample)",
  markov_cross_transfer: "Classical Markov transfer",
  qcbm_cross_transfer: "Quantum (QCBM) transfer",
  gan_cross_transfer: "Classical GAN transfer",
};

const METHOD_COLORS = {
  no_augmentation: "#94A3B8",
  bootstrap: "#F59E0B",
  markov_cross_transfer: "#3B82F6",
  qcbm_cross_transfer: "#10B981",
  gan_cross_transfer: "#8B5CF6",
};

function Card({ title, subtitle, children }) {
  return (
    <div style={{
      background: "#FFFFFF",
      borderRadius: 12,
      padding: 24,
      marginBottom: 24,
      boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
      border: "1px solid #E5E7EB",
    }}>
      <h2 style={{ fontSize: 18, margin: 0, color: "#111827" }}>{title}</h2>
      {subtitle && <p style={{ color: "#6B7280", fontSize: 14, marginTop: 4, marginBottom: 16 }}>{subtitle}</p>}
      {children}
    </div>
  );
}

function SummaryChart({ data }) {
  if (!data) return <p>Loading...</p>;
  const chartData = data.map((d) => ({
    ...d,
    label: METHOD_LABELS[d.method] || d.method,
  }));
  return (
    <ResponsiveContainer width="100%" height={340}>
      <BarChart data={chartData} margin={{ bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
        <XAxis dataKey="label" stroke="#6B7280" tick={{ fontSize: 12 }} angle={-15} textAnchor="end" interval={0} height={60} />
        <YAxis stroke="#6B7280" domain={[0, "auto"]} />
        <Tooltip contentStyle={{ background: "#FFFFFF", border: "1px solid #E5E7EB", borderRadius: 8 }} />
        <Legend />
        <Bar dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]}>
          {chartData.map((d, i) => <Bar key={i} />)}
        </Bar>
        <Bar dataKey="macro_f1" name="Macro F1" fill="#C4B5FD" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function FilterBar({ options, filters, setFilters }) {
  const selectStyle = {
    padding: "8px 12px",
    borderRadius: 8,
    border: "1px solid #D1D5DB",
    background: "#FFFFFF",
    fontSize: 14,
    marginRight: 12,
    marginBottom: 8,
  };
  return (
    <div style={{ marginBottom: 16 }}>
      <select style={selectStyle} value={filters.method} onChange={(e) => setFilters({ ...filters, method: e.target.value })}>
        <option value="">All methods</option>
        {options.methods.map((m) => <option key={m} value={m}>{METHOD_LABELS[m] || m}</option>)}
      </select>
      <select style={selectStyle} value={filters.sector} onChange={(e) => setFilters({ ...filters, sector: e.target.value })}>
        <option value="">All sector matches</option>
        {options.sectors.map((s) => <option key={s} value={s}>{s}</option>)}
      </select>
      <select style={selectStyle} value={filters.vol} onChange={(e) => setFilters({ ...filters, vol: e.target.value })}>
        <option value="">All volatility regimes</option>
        {options.vols.map((v) => <option key={v} value={v}>{v}</option>)}
      </select>
      {(filters.method || filters.sector || filters.vol) && (
        <button
          onClick={() => setFilters({ method: "", sector: "", vol: "" })}
          style={{ ...selectStyle, cursor: "pointer", color: "#6B7280" }}
        >
          Clear filters
        </button>
      )}
    </div>
  );
}

function ConditionsTable({ data }) {
  const [filters, setFilters] = useState({ method: "", sector: "", vol: "" });
  const [sortKey, setSortKey] = useState("accuracy");
  const [sortDir, setSortDir] = useState("desc");

  const options = useMemo(() => {
    if (!data) return { methods: [], sectors: [], vols: [] };
    return {
      methods: [...new Set(data.map((d) => d.method))],
      sectors: [...new Set(data.map((d) => d.sector_match).filter(Boolean))],
      vols: [...new Set(data.map((d) => d.source_vol_bucket).filter(Boolean))],
    };
  }, [data]);

  const filtered = useMemo(() => {
    if (!data) return [];
    let rows = data.filter((d) =>
      (!filters.method || d.method === filters.method) &&
      (!filters.sector || d.sector_match === filters.sector) &&
      (!filters.vol || d.source_vol_bucket === filters.vol)
    );
    rows.sort((a, b) => {
      const av = a[sortKey] ?? -Infinity;
      const bv = b[sortKey] ?? -Infinity;
      return sortDir === "asc" ? av - bv : bv - av;
    });
    return rows;
  }, [data, filters, sortKey, sortDir]);

  if (!data) return <p>Loading...</p>;

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir(sortDir === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("desc"); }
  };

  const th = (key, label) => (
    <th
      onClick={() => toggleSort(key)}
      style={{ padding: 10, cursor: "pointer", userSelect: "none", color: "#374151", fontSize: 13, borderBottom: "2px solid #E5E7EB" }}
    >
      {label} {sortKey === key ? (sortDir === "asc" ? "\u2191" : "\u2193") : ""}
    </th>
  );

  return (
    <div>
      <FilterBar options={options} filters={filters} setFilters={setFilters} />
      <p style={{ fontSize: 13, color: "#6B7280", marginBottom: 12 }}>
        Showing {filtered.length} of {data.length} rows. Click a column header to sort.
      </p>
      <div style={{ overflowX: "auto", maxHeight: 500, overflowY: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead style={{ position: "sticky", top: 0, background: "#F9FAFB" }}>
            <tr>
              <th style={{ padding: 10, textAlign: "left", color: "#374151" }}>Source</th>
              <th style={{ padding: 10, textAlign: "left", color: "#374151" }}>Target</th>
              <th style={{ padding: 10, textAlign: "left", color: "#374151" }}>Method</th>
              <th style={{ padding: 10, textAlign: "left", color: "#374151" }}>Sector</th>
              <th style={{ padding: 10, textAlign: "left", color: "#374151" }}>Vol regime</th>
              {th("accuracy", "Accuracy")}
              {th("macro_f1", "Macro F1")}
            </tr>
          </thead>
          <tbody>
            {filtered.map((row, i) => (
              <tr key={i} style={{ borderBottom: "1px solid #F3F4F6" }}>
                <td style={{ padding: 10 }}>{row.source}</td>
                <td style={{ padding: 10 }}>{row.target}</td>
                <td style={{ padding: 10 }}>
                  <span style={{
                    background: (METHOD_COLORS[row.method] || "#94A3B8") + "22",
                    color: METHOD_COLORS[row.method] || "#374151",
                    padding: "2px 8px",
                    borderRadius: 999,
                    fontSize: 12,
                  }}>
                    {METHOD_LABELS[row.method] || row.method}
                  </span>
                </td>
                <td style={{ padding: 10 }}>{row.sector_match ?? "-"}</td>
                <td style={{ padding: 10 }}>{row.source_vol_bucket ?? "-"}</td>
                <td style={{ padding: 10, fontWeight: 600 }}>{row.accuracy?.toFixed(4)}</td>
                <td style={{ padding: 10 }}>{row.macro_f1?.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function App() {
  const summary = useApi("/api/augmentation-summary");
  const conditions = useApi("/api/conditions");

  return (
    <div style={{ background: "#F3F4F6", minHeight: "100vh", fontFamily: "system-ui, -apple-system, sans-serif" }}>
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px" }}>
        <h1 style={{ fontSize: 28, color: "#111827", marginBottom: 4 }}>QCBM Cross-Transfer Dashboard</h1>
        <p style={{ color: "#6B7280", marginBottom: 32, fontSize: 15 }}>
          Testing whether synthetic training data from data-rich large-cap stocks
          (generated via a quantum circuit, a classical Markov chain, bootstrap
          resampling, or a classical GAN) improves next-day prediction accuracy
          for data-scarce small-cap stocks.
        </p>

        <Card
          title="Mean accuracy by augmentation method"
          subtitle="Averaged across all small-cap stocks. Higher is better. 'No augmentation' is the control — beating it is what would justify using synthetic data at all."
        >
          {summary.error && <p style={{ color: "#DC2626" }}>Error: {summary.error}</p>}
          <SummaryChart data={summary.data} />
        </Card>

        <Card
          title="Per source \u2192 target pair, filterable"
          subtitle="Every large-cap (source) \u2192 small-cap (target) combination tested, with sector match and volatility regime. Use the filters to explore when transfer helps vs. hurts."
        >
          {conditions.error && <p style={{ color: "#DC2626" }}>Error: {conditions.error}</p>}
          <ConditionsTable data={conditions.data} />
        </Card>
      </div>
    </div>
  );
}
