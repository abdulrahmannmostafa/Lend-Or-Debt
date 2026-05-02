import { useState } from "react";
import { API, post } from "../api";
import { Spinner } from "./Spinner";
import { PlotImg } from "./PlotImg";

const EDA_TABS = [
  { id: "univariate",             label: "Univariate" },
  { id: "pie",                    label: "Pie Chart" },
  { id: "continuous",             label: "Continuous vs Continuous" },
  { id: "discrete_vs_continuous", label: "Discrete vs Continuous" },
  { id: "discrete_vs_target",     label: "Discrete vs Target" },
  { id: "correlation",            label: "Correlation Matrix" },
  { id: "dashboard_with_smote",     label: "Dashboard with SMOTE" },
  { id: "dashboard_without_smote",     label: "Dashboard without SMOTE" }
];

export function EDASection() {
  const [ready, setReady] = useState(false);
  const [initLoading, setInitLoading] = useState(false);
  const [initMsg, setInitMsg] = useState(null);
  const [columns, setColumns] = useState({ cleaned: [], transformed: [], discrete_features: [] });

  const [tab, setTab] = useState("univariate");
  const [loading, setLoading] = useState(false);
  const [plotImg, setPlotImg] = useState(null);
  const [error, setError] = useState(null);

  // per-tab params
  const [column,   setColumn]   = useState("");
  const [dataType, setDataType] = useState(0); // 0=transformed 1=cleaned 2=transformed without smote
  const [target,   setTarget]   = useState("");

  const initEDA = async () => {
    setInitLoading(true);
    const res = await post(`${API}/eda/init`, {});
    setInitLoading(false);
    if (res.error) { setInitMsg(res.error); return; }
    setInitMsg(res.message);
    setReady(true);
    const cols = await fetch(`${API}/eda/columns`).then((r) => r.json());
    if (!cols.error) setColumns(cols);
  };


  const colList = dataType === 1 ? columns.cleaned : columns.transformed;

  const runEDA = async () => {
    setLoading(true);
    setPlotImg(null);
    setError(null);

    let body = {};
    if (tab === "univariate")                  body = { column, data_type: dataType };
    else if (tab === "pie")                    body = { feature: column, data_type: dataType };
    else if (tab === "discrete_vs_target")     body = { target_column: target };
    else if (tab === "discrete_vs_continuous") body = { feature: column };
    // correlation needs no params

    const res = await post(`${API}/eda/${tab}`, body);
    setLoading(false);
    if (res.error) setError(res.error);
    else setPlotImg(res.image);
  };

  return (
    <div className="card eda-card">
      <h2>📊 EDA</h2>

      {!ready ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <p>Initialise the EDA module — loads cleaned &amp; transformed data from default paths.</p>
          <button className="btn-dark" onClick={initEDA} disabled={initLoading}>
            {initLoading ? <Spinner /> : "Initialise EDA"}
          </button>
          {initMsg && <p className={initMsg.includes("ready") ? "ok" : "err"}>{initMsg}</p>}
        </div>
      ) : (
        <>
          <div className="tabs">
            {EDA_TABS.map((t) => (
              <button
                key={t.id}
                className={`tab ${tab === t.id ? "active" : ""}`}
                onClick={() => { setTab(t.id); setPlotImg(null); setError(null); }}
              >
                {t.label}
              </button>
            ))}
          </div>

          {(tab === "univariate" || tab === "pie") && (
            <div className="row">
              <label>Data</label>
              <div className="seg">
                <button className={dataType === 0 ? "active" : ""} onClick={() => setDataType(0)}>transformed</button>
                <button className={dataType === 1 ? "active" : ""} onClick={() => setDataType(1)}>cleaned</button>
                <button className={dataType === 2 ? "active" : ""} onClick={() => setDataType(2)}>transformed without smote</button>
              </div>
            </div>
          )}

          {(tab === "univariate") && (
            <div className="row">
              <label>Column</label>
              <select value={column} onChange={(e) => setColumn(e.target.value)}>
                <option value="">— pick column —</option>
                {colList.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          )}

          {(tab === "pie") && (
            <div className="row">
              <label>Column</label>
              <select value={column} onChange={(e) => setColumn(e.target.value)}>
                <option value="">— pick column —</option>
                {columns.mapping
                  .filter((c) => colList.includes(c))
                  .map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>

          )}

          {(tab === "discrete_vs_continuous") && (
            <div className="row">
              <label>Column</label>
              <select value={column} onChange={(e) => setColumn(e.target.value)}>
                <option value="">— pick column —</option>
                {columns.continuous_features
                  .map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>

          )}

          {tab === "discrete_vs_target" && (
            <div className="row">
              <label>Target</label>
              <select value={target} onChange={(e) => setTarget(e.target.value)}>
                <option value="">— pick target —</option>
                {columns.discrete_features.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          )}

          <button className="btn-dark" onClick={runEDA} disabled={loading}>
            {loading ? <Spinner /> : "▶ Run"}
          </button>

          {error && <p className="err">✗ {error}</p>}
          <PlotImg src={plotImg} />
        </>
      )}
    </div>
  );
}
