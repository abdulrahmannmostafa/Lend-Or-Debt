export function LogBox({ stdout, stderr, success }) {
  if (!stdout && !stderr) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {success !== undefined && (
        <p className={success ? "ok" : "err"}>{success ? "✓ Completed" : "✗ Failed"}</p>
      )}
      {stdout && <pre className="log">{stdout}</pre>}
      {stderr && <pre className="log log-err">{stderr}</pre>}
    </div>
  );
}
