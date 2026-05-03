export const globalStyles = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #f5f5f5; color: #111; font-family: system-ui, sans-serif; font-size: 14px; }

  .top-bar { padding: 14px 24px; background: #fff; border-bottom: 1px solid #ddd; }
  .top-bar h1 { font-size: 1rem; font-weight: 600; }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 20px 24px; max-width: 1200px; margin: 0 auto; }
  .eda-card { grid-column: 1 / -1; }

  .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  h2 { font-size: .95rem; font-weight: 600; }
  p { color: #555; line-height: 1.5; }
  label { font-size: .82rem; color: #555; }

  .row { display: flex; align-items: center; gap: 10px; }
  select, input[type=text] { flex: 1; padding: 7px 10px; border: 1px solid #ccc; border-radius: 6px; font-size: .85rem; background: #fff; }
  select:focus, input:focus { outline: none; border-color: #555; }

  .drop-zone { border: 1.5px dashed #ccc; border-radius: 6px; padding: 20px; text-align: center; color: #888; cursor: pointer; }
  .drop-zone:hover { border-color: #888; background: #fafafa; }

  button { padding: 8px 16px; border: none; border-radius: 6px; font-size: .85rem; font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }
  button:disabled { opacity: .4; cursor: not-allowed; }
  .btn-dark   { background: #111; color: #fff; }
  .btn-dark:hover:not(:disabled) { background: #333; }

  .tabs { display: flex; gap: 4px; flex-wrap: wrap; border-bottom: 1px solid #ddd; padding-bottom: 8px; }
  .tab { padding: 5px 12px; border: 1px solid #ddd; border-radius: 6px; background: #fff; font-size: .8rem; cursor: pointer; color: #555; }
  .tab.active { background: #111; color: #fff; border-color: #111; }

  .seg { display: flex; border: 1px solid #ccc; border-radius: 6px; overflow: hidden; }
  .seg button { border-radius: 0; border: none; background: #fff; color: #555; font-weight: 400; padding: 6px 12px; font-size: .8rem; }
  .seg button:not(:last-child) { border-right: 1px solid #ccc; }
  .seg button.active { background: #111; color: #fff; }

  .log { background: #f8f8f8; border: 1px solid #ddd; border-radius: 6px; padding: 10px; font-family: monospace; font-size: .78rem; white-space: pre-wrap; max-height: 240px; overflow-y: auto; color: #333; }
  .log-err { color: #c00; }
  .ok  { color: #1a7f1a; font-weight: 600; font-size: .85rem; }
  .err { color: #c00; font-size: .85rem; }

  .plot-img { width: 100%; border-radius: 6px; border: 1px solid #ddd; }

  .spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid rgba(255,255,255,.3); border-top-color: #fff; border-radius: 50%; animation: spin .6s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  @media(max-width: 800px) { .grid { grid-template-columns: 1fr; padding: 12px; } .eda-card { grid-column: 1; } }
`;

export const introStyles = {
  section: {
    padding: '30px 20px',
    background: '#f5f5f5',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  container: {
    maxWidth: '600px',
    textAlign: 'center',
  },
  title: {
    fontSize: '32px',
    fontWeight: '600',
    marginBottom: '16px',
    margin: '0 0 16px 0',
  },
  paragraph: {
    fontSize: '14px',
    lineHeight: '1.6',
    marginBottom: '20px',
    color: '#555',
  },
  image: {
    maxWidth: '100%',
    height: 'auto',
    borderRadius: '6px',
  },
};
