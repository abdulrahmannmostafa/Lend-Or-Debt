import sys
import os
import io
import base64
import subprocess
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
from flask_cors import CORS
from src.pipeline.eda import EDA 

# ── path setup: add project root so src.pipeline.* is importable ──────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

_eda_instance: EDA | None = None


# ── helpers ───────────────────────────────────────────────────────────────────
def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def capture_fig() -> str:
    """Grab whatever plt just drew, encode as base64, then close."""
    fig = plt.gcf()
    return fig_to_base64(fig)


# ── UPLOAD ────────────────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    # dest is a relative path inside the project, e.g. "data/clean/train_cleaned.csv"
    dest = request.form.get("dest", "")
    if not dest:
        return jsonify({"error": "Must provide 'dest' — relative path inside project root"}), 400

    save_path = os.path.join(PROJECT_ROOT, dest)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)

    return jsonify({"message": f"Saved to {dest}", "path": save_path})


# ── CLEANING — runs __main__ of data_cleaning.py ──────────────────────────────
@app.route("/api/run/cleaning", methods=["POST"])
def run_cleaning():
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline.data_cleaning"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return jsonify({
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    })


# ── TRANSFORMATION — runs __main__ of data_transformation.py ─────────────────
@app.route("/api/run/transformation", methods=["POST"])
def run_transformation():
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline.data_transformation"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return jsonify({
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    })


# ── EDA — initialise the EDA instance ────────────────────────────────────────
@app.route("/api/eda/init", methods=["POST"])
def eda_init():
    global _eda_instance
    body = request.json or {}

    def abs_path(key, default):
        p = body.get(key, default)
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    _eda_instance = EDA(
        train_input_cleaned=abs_path("train_cleaned",     "data/clean/train_cleaned.csv"),
        val_input_cleaned=abs_path("val_cleaned",         "data/clean/val_cleaned.csv"),
        test_input_cleaned=abs_path("test_cleaned",       "data/clean/test_cleaned.csv"),
        train_input_transformed=abs_path("train_transformed", "data/transformed/train_transformed.csv"),
        val_input_transformed=abs_path("val_transformed",     "data/transformed/val_transformed.csv"),
        test_input_transformed=abs_path("test_transformed",   "data/transformed/test_transformed.csv"),
    )
    _eda_instance.load_data_cleaned()
    _eda_instance.load_data_transformed()

    return jsonify({"message": "EDA instance ready"})


def _require_eda():
    if _eda_instance is None:
        return None, (jsonify({"error": "EDA not initialised. Call /api/eda/init first."}), 400)
    return _eda_instance, None


# ── EDA: apply_univariate ─────────────────────────────────────────────────────
@app.route("/api/eda/univariate", methods=["POST"])
def eda_univariate():
    eda, err = _require_eda()
    if err: return err

    body      = request.json or {}
    column    = body.get("column")
    data_type = int(body.get("data_type", 0))   # 0=transformed, 1=cleaned

    if not column:
        return jsonify({"error": "column is required"}), 400

    plt.close("all")
    eda.apply_univariate(column_name=column, data_type=data_type)
    return jsonify({"image": capture_fig()})


# ── EDA: apply_pie_chart ──────────────────────────────────────────────────────
@app.route("/api/eda/pie", methods=["POST"])
def eda_pie():
    eda, err = _require_eda()
    if err: return err

    body      = request.json or {}
    feature   = body.get("feature")
    data_type = int(body.get("data_type", 0))

    if not feature:
        return jsonify({"error": "feature is required"}), 400

    # use EDA's own mapping if available, else identity
    mapping = eda.mapping.get(feature)
    if mapping is None:
        df = eda.full_df_cleaned if data_type else eda.full_df_transformed
        mapping = {v: str(v) for v in sorted(df[feature].dropna().unique())}

    plt.close("all")
    eda.apply_pie_chart(feature=feature, mapping=mapping, data_type=data_type)
    return jsonify({"image": capture_fig()})


# ── EDA: continues_versus_continuous_eda ──────────────────────────────────────
@app.route("/api/eda/continuous", methods=["POST"])
def eda_continuous():
    eda, err = _require_eda()
    if err: return err

    body     = request.json or {}
    features = body.get("features")

    if not features or not isinstance(features, list):
        return jsonify({"error": "features must be a non-empty list of column names"}), 400

    plt.close("all")
    eda.continues_versus_continuous_eda(continuous_features=features)
    return jsonify({"image": capture_fig()})


# ── EDA: discrete_versus_continuous_eda ───────────────────────────────────────
@app.route("/api/eda/discrete_vs_continuous", methods=["POST"])
def eda_discrete_vs_continuous():
    eda, err = _require_eda()
    if err: return err

    body    = request.json or {}
    feature = body.get("feature")

    if not feature:
        return jsonify({"error": "feature is required"}), 400

    plt.close("all")
    eda.continuous_versus_discrete_eda(feature=feature)
    return jsonify({"image": capture_fig()})


# ── EDA: discrete_versus_target_stacked ──────────────────────────────────────
@app.route("/api/eda/discrete_vs_target", methods=["POST"])
def eda_discrete_vs_target():
    eda, err = _require_eda()
    if err: return err

    body          = request.json or {}
    target_column = body.get("target_column")

    if not target_column:
        return jsonify({"error": "target_column is required"}), 400

    plt.close("all")
    eda.discrete_versus_target_stacked(target_column=target_column)
    return jsonify({"image": capture_fig()})


# ── EDA: plot_large_correlation_matrix ────────────────────────────────────────
@app.route("/api/eda/correlation", methods=["POST"])
def eda_correlation():
    eda, err = _require_eda()
    if err: return err

    plt.close("all")
    eda.plot_large_correlation_matrix()
    return jsonify({"image": capture_fig()})


# ── EDA: columns — utility for frontend dropdowns ─────────────────────────────
@app.route("/api/eda/columns", methods=["GET"])
def eda_columns():
    eda, err = _require_eda()
    if err: return err

    return jsonify({
        "cleaned":          list(eda.full_df_cleaned.columns)     if eda.full_df_cleaned     is not None else [],
        "transformed":      list(eda.full_df_transformed.columns) if eda.full_df_transformed is not None else [],
        "discrete_features": eda.discrete_features,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
