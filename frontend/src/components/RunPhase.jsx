import { useState } from "react";
import { API, post } from "../api";
import { Spinner } from "./Spinner";
import { LogBox } from "./LogBox";

export function RunPhase({ title, icon, endpoint, desc }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const run = async () => {
    setLoading(true);
    setResult(null);
    const res = await post(`${API}/run/${endpoint}`, {});
    setLoading(false);
    setResult(res);
  };

  return (
    <div className="card">
      <h2>{icon} {title}</h2>
      <p>{desc}</p>
      <button className="btn-dark" onClick={run} disabled={loading}>
        {loading ? <Spinner /> : "▶ Run"}
      </button>
      {result && (
        <LogBox stdout={result.stdout} stderr={result.stderr} success={result.success} />
      )}
    </div>
  );
}
