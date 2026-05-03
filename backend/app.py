import sys
import os

# ── path setup: add project root so src.pipeline.* is importable ──────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
import io
import base64
import logging
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
from flask_cors import CORS
from src.pipeline.eda import EDA
from src.pipeline.data_cleaning import clean_data
from src.pipeline.data_transformation import run_transformation


app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

_eda_instance: EDA | None = None

# ── tracks the uploaded raw input path so cleaning/transform use it ───────────
_uploaded_input_path: str | None = None


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
    global _uploaded_input_path

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    # Save the raw uploaded CSV into data/raw/ inside the project
    save_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)
    file.save(save_path)

    # Remember the path so cleaning/transformation use it
    _uploaded_input_path = save_path
    log.info("Uploaded raw file saved to: %s", save_path)

    return jsonify({"message": f"Uploaded '{file.filename}' successfully.", "path": save_path})


# ── CLEANING — calls clean_data() directly with the uploaded file path ─────────
@app.route("/api/run/cleaning", methods=["POST"])
def run_cleaning():
    global _uploaded_input_path

    # Use uploaded file if available, otherwise fall back to the default path
    input_path = _uploaded_input_path or os.path.join(PROJECT_ROOT, "data", "taiwan_merged.csv")

    if not os.path.exists(input_path):
        return jsonify({
            "returncode": 1,
            "stdout": "",
            "stderr": f"Input file not found: {input_path}. Please upload a CSV first.",
            "success": False,
        }), 400

    train_out = os.path.join(PROJECT_ROOT, "data", "clean", "train_cleaned.csv")
    val_out   = os.path.join(PROJECT_ROOT, "data", "clean", "val_cleaned.csv")
    test_out  = os.path.join(PROJECT_ROOT, "data", "clean", "test_cleaned.csv")

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    logging.getLogger().addHandler(handler)

    try:
        clean_data(
            input_path=input_path,
            train_output=train_out,
            val_output=val_out,
            test_output=test_out,
        )
        logging.getLogger().removeHandler(handler)
        return jsonify({
            "returncode": 0,
            "stdout": log_capture.getvalue(),
            "stderr": "",
            "success": True,
        })
    except Exception:
        logging.getLogger().removeHandler(handler)
        return jsonify({
            "returncode": 1,
            "stdout": log_capture.getvalue(),
            "stderr": traceback.format_exc(),
            "success": False,
        }), 500



#-- debug

@app.route("/api/debug", methods=["GET"])
def debug():
    return jsonify({
        "PROJECT_ROOT": PROJECT_ROOT,
        "uploaded_input_path": _uploaded_input_path,
        "clean_train_exists": os.path.exists(os.path.join(PROJECT_ROOT, "data/clean/train_cleaned.csv")),
        "transformed_train_exists": os.path.exists(os.path.join(PROJECT_ROOT, "data/transformed/train_transformed.csv")),
        "eda_initialised": _eda_instance is not None,
        "eda_train_shape": str(_eda_instance.full_df_cleaned.shape) if _eda_instance and _eda_instance.full_df_cleaned is not None else None,
    })


# ── TRANSFORMATION — calls run_transformation() directly with cleaned paths ────
@app.route("/api/run/transformation", methods=["POST"])
def run_transformation_route():
    train_in = os.path.join(PROJECT_ROOT, "data", "clean", "train_cleaned.csv")
    val_in   = os.path.join(PROJECT_ROOT, "data", "clean", "val_cleaned.csv")
    test_in  = os.path.join(PROJECT_ROOT, "data", "clean", "test_cleaned.csv")

    missing = [p for p in [train_in, val_in, test_in] if not os.path.exists(p)]
    if missing:
        return jsonify({
            "returncode": 1,
            "stdout": "",
            "stderr": f"Cleaned files not found: {missing}. Run Data Cleaning first.",
            "success": False,
        }), 400

    train_out = os.path.join(PROJECT_ROOT, "data", "transformed", "train_transformed.csv")
    val_out   = os.path.join(PROJECT_ROOT, "data", "transformed", "val_transformed.csv")
    test_out  = os.path.join(PROJECT_ROOT, "data", "transformed", "test_transformed.csv")

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    logging.getLogger().addHandler(handler)

    try:
        run_transformation(
            train_input=train_in,
            val_input=val_in,
            test_input=test_in,
            train_output=train_out,
            val_output=val_out,
            test_output=test_out,
        )
        logging.getLogger().removeHandler(handler)
        return jsonify({
            "returncode": 0,
            "stdout": log_capture.getvalue(),
            "stderr": "",
            "success": True,
        })
    except Exception:
        logging.getLogger().removeHandler(handler)
        return jsonify({
            "returncode": 1,
            "stdout": log_capture.getvalue(),
            "stderr": traceback.format_exc(),
            "success": False,
        }), 500


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
        train_input_transformed_without_smote=abs_path("train_transformed_without_smote", "data/transformed/train_transformed_without_smote.csv"),
        val_input_transformed_without_smote=abs_path("val_transformed_without_smote",     "data/transformed/val_transformed_without_smote.csv"),
        test_input_transformed_without_smote=abs_path("test_transformed_without_smote",   "data/transformed/test_transformed_without_smote.csv"),
    )
    _eda_instance.load_data_cleaned()
    _eda_instance.load_data_transformed()
    _eda_instance.load_data_transformed_without_smote()

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
    data_type = int(body.get("data_type", 0))

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

    mapping = eda.mapping.get(feature)

    if isinstance(mapping, dict):
        pie_mapping = mapping
    elif isinstance(mapping, list):
        if feature == "default payment next month":
            pie_mapping = {0: mapping[0], 1: mapping[1]}
        elif feature == "SEX":
            pie_mapping = {1: mapping[0], 2: mapping[1]}
        elif feature == "MARRIAGE":
            pie_mapping = {1: mapping[0], 2: mapping[1], 3: mapping[2]}
        elif feature == "EDUCATION":
            pie_mapping = {1: mapping[0], 2: mapping[1], 3: mapping[2], 4: mapping[3]}
        elif feature == "is_anomaly":
            pie_mapping = {0: mapping[0], 1: mapping[1]}
        elif feature == "is_underpaying":
            pie_mapping = {0: mapping[0], 1: mapping[1]}
        else:
            pie_mapping = {i: v for i, v in enumerate(mapping)}
    else:
        df = eda.full_df_cleaned if data_type == 1 else eda.full_df_transformed
        pie_mapping = {v: str(v) for v in sorted(df[feature].dropna().unique())}

    plt.close("all")
    eda.apply_pie_chart(feature=feature, mapping=pie_mapping, data_type=data_type)
    return jsonify({"image": capture_fig()})


# ── EDA: continues_versus_continuous_eda ──────────────────────────────────────
@app.route("/api/eda/continuous", methods=["POST"])
def eda_continuous():
    eda, err = _require_eda()
    if err: return err

    body = request.json or {}
    feature_1 = body.get("feature_1")
    feature_2 = body.get("feature_2")

    if not feature_1 or not feature_2:
        transformed = eda.full_df_transformed.columns if eda.full_df_transformed is not None else []
        available_features = [c for c in transformed if c not in eda.discrete_features]
        if len(available_features) < 2:
            return jsonify({"error": "Need at least two continuous features"}), 400
        feature_1, feature_2 = available_features[:2]

    plt.close("all")
    eda.continuous_vs_continuous_eda(feature_1, feature_2)
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
        "cleaned":           list(eda.full_df_cleaned.columns)     if eda.full_df_cleaned     is not None else [],
        "transformed":       list(eda.full_df_transformed.columns) if eda.full_df_transformed is not None else [],
        "discrete_features": eda.discrete_features,
        "mapping":           list(eda.mapping.keys()),
        "continuous_features": eda.continuous_features,
    })


# ── EDA: dashboard_with_smote ─────────────────────────────────────────────────
@app.route("/api/eda/dashboard_with_smote", methods=["POST"])
def eda_dashboard_with_smote():
    image_path = os.path.join(PROJECT_ROOT, "dashboard_with_smotes.png")

    if not os.path.exists(image_path):
        return jsonify({"error": f"Image not found at {image_path}"}), 404

    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")

    return jsonify({"image": encoded})


# ── EDA: dashboard_without_smote ──────────────────────────────────────────────
@app.route("/api/eda/dashboard_without_smote", methods=["POST"])
def eda_dashboard_without_smote():
    image_path = os.path.join(PROJECT_ROOT, "dashboard_without_smotes.png")

    if not os.path.exists(image_path):
        return jsonify({"error": f"Image not found at {image_path}"}), 404

    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")

    return jsonify({"image": encoded})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)