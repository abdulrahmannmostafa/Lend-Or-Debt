import { useState } from "react";
import { API } from "../api";
import { Spinner } from "./Spinner";

const RAW_INPUT_PATH = "data/taiwan_merged1.csv";

export function UploadSection() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const upload = async () => {
    if (!file) return;
    setLoading(true);
    const fd = new FormData();
    fd.append("file", file);
    fd.append("dest", RAW_INPUT_PATH);
    const res = await fetch(`${API}/upload`, { method: "POST", body: fd }).then((r) => r.json());
    setLoading(false);
    setResult(res);
  };

  return (
    <div className="card">
      <h2>📂 Upload Raw Dataset</h2>
      <p>
        Upload the raw CSV to run cleaning and transformation on. It will be saved as{" "}
        <code style={{ background: "#f3f3f3", padding: "1px 5px", borderRadius: 4, fontSize: ".8rem" }}>
          {RAW_INPUT_PATH}
        </code>.
      </p>

      <div
        className="drop-zone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          setFile(e.dataTransfer.files[0]);
          setResult(null);
        }}
      >
        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          onChange={(e) => { setFile(e.target.files[0]); setResult(null); }}
          id="fu"
          style={{ display: "none" }}
        />
        <label htmlFor="fu">{file ? `📄 ${file.name}` : "⬆️ Drop CSV or click to browse"}</label>
      </div>

      <button className="btn-dark" onClick={upload} disabled={!file || loading}>
        {loading ? <Spinner /> : "Upload"}
      </button>

      {result && (
        <p className={result.error ? "err" : "ok"}>
          {result.error || `✓ ${result.message} — ready to clean & transform.`}
        </p>
      )}
    </div>
  );
}
