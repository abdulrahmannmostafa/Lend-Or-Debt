export function PlotImg({ src }) {
  if (!src) return null;
  return (
    <img
      src={`data:image/png;base64,${src}`}
      alt="plot"
      className="plot-img"
    />
  );
}
