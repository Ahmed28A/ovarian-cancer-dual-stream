"""
Ovarian Cancer Dual-Stream — interactive demo (Gradio).

Tab 1  Clinical (Stream B): CatBoost on 16 blood biomarkers + per-patient SHAP explanation.
       The model is trained at start-up from data/Supplementary data 1.xlsx (takes a few seconds).
Tab 2  Histopathology (Stream A): ConvNeXt-Base + CBAM subtype classifier + Grad-CAM.
       Needs the trained checkpoint at weights/stream_a_convnext_single_cbam.pth.

Run locally:   pip install -r requirements.txt  &&  python app.py
Deploy:        upload this folder (plus data/ and weights/) to a Hugging Face Space (Gradio SDK).

Research demo only. Not a medical device and not for clinical decisions.
"""
import os
import re
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SEED = 42


def _first_existing(*paths):
    for p in paths:
        if p.exists():
            return p
    return paths[0]


DATA_PATH = Path(os.environ.get("CLINICAL_DATA", "")) if os.environ.get("CLINICAL_DATA") else _first_existing(
    HERE / "data" / "Supplementary data 1.xlsx", HERE.parent / "data" / "Supplementary data 1.xlsx"
)
WEIGHTS_PATH = Path(os.environ.get("STREAM_A_WEIGHTS", "")) if os.environ.get("STREAM_A_WEIGHTS") else _first_existing(
    HERE / "weights" / "stream_a_convnext_single_cbam.pth",
    HERE.parent / "weights" / "stream_a_convnext_single_cbam.pth",
)

# 16 features selected by Mutual Information in notebooks/02_stream_b_clinical_catboost_shap.ipynb
FEATURES = ["AFP", "AG", "Age", "ALB", "ALP", "Ca", "CA125", "CEA",
            "HE4", "LYM%", "MCH", "Menopause", "Na", "NEU", "PDW", "PLT"]
FEATURE_LABELS = {
    "AFP": "AFP (alpha-fetoprotein)", "AG": "AG (anion gap)", "Age": "Age (years)",
    "ALB": "ALB (albumin)", "ALP": "ALP (alkaline phosphatase)", "Ca": "Ca (calcium)",
    "CA125": "CA-125", "CEA": "CEA (carcinoembryonic antigen)", "HE4": "HE4",
    "LYM%": "LYM% (lymphocyte %)", "MCH": "MCH", "Menopause": "Menopause (0 = no, 1 = yes)",
    "Na": "Na (sodium)", "NEU": "NEU (neutrophils)", "PDW": "PDW (platelet distribution width)",
    "PLT": "PLT (platelets)",
}
# Optuna (leak-free, 5-fold CV) best parameters from the notebook
CATBOOST_PARAMS = dict(iterations=316, learning_rate=0.06967890649270345, depth=6,
                       l2_leaf_reg=2.344442122896348, bagging_temperature=0.0029019809570722307,
                       random_seed=SEED, verbose=False)


def _clean_numeric(val):
    if pd.isna(val):
        return np.nan
    try:
        return float(re.sub(r"[><=\s\t]", "", str(val)))
    except ValueError:
        return np.nan


# ───────────────────────────── Stream B: clinical ─────────────────────────────
class ClinicalModel:
    def __init__(self, path):
        df = pd.read_excel(path)
        for col in FEATURES:
            if df[col].dtype == object:
                df[col] = df[col].apply(_clean_numeric)
        X, y = df[FEATURES].astype(float), df["TYPE"].astype(int).values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

        self.medians = X_tr.median()
        self.imputer = SimpleImputer(strategy="median").fit(X_tr)
        self.scaler = StandardScaler().fit(self.imputer.transform(X_tr))
        Xs = self.scaler.transform(self.imputer.transform(X_tr))
        Xs, y_res = SMOTE(random_state=SEED).fit_resample(Xs, y_tr)
        self.model = CatBoostClassifier(**CATBOOST_PARAMS).fit(Xs, y_res)

        Xt = self._prep(X_te)
        proba = self.model.predict_proba(Xt)[:, 1]
        pred = (proba >= 0.5).astype(int)
        self.metrics = dict(acc=accuracy_score(y_te, pred) * 100, recall=recall_score(y_te, pred) * 100,
                            auc=roc_auc_score(y_te, proba) * 100, n=len(y_te))

    def _prep(self, X):
        return self.scaler.transform(self.imputer.transform(X))

    def predict(self, values):
        X = pd.DataFrame([values], columns=FEATURES).astype(float)
        Xs = self._prep(X)
        p = float(self.model.predict_proba(Xs)[0, 1])
        shap_vals = self.model.get_feature_importance(Pool(Xs), type="ShapValues")[0, :-1]
        return p, shap_vals


CLINICAL = None
CLINICAL_ERROR = None
try:
    CLINICAL = ClinicalModel(DATA_PATH)
except Exception as e:  # dataset missing or unreadable
    CLINICAL_ERROR = f"Clinical model not available: {e}. Put 'Supplementary data 1.xlsx' in data/ (see data/README.md)."


def _shap_plot(shap_vals, values):
    order = np.argsort(np.abs(shap_vals))[::-1][:10][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
    colors = ["#E84040" if shap_vals[i] > 0 else "#4A90D9" for i in order]
    ax.barh([f"{FEATURES[i]} = {values[i]:g}" for i in order], shap_vals[order], color=colors)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_xlabel("SHAP value (red pushes toward cancer, blue toward benign)")
    ax.set_title("Why the model decided this (top 10 features)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def predict_clinical(*vals):
    if CLINICAL is None:
        raise gr.Error(CLINICAL_ERROR)
    values = [float(v) if v is not None else float(CLINICAL.medians[f]) for v, f in zip(vals, FEATURES)]
    p, shap_vals = CLINICAL.predict(values)
    return {"Malignant (ovarian cancer)": p, "Benign ovarian tumour": 1 - p}, _shap_plot(shap_vals, values)


# ───────────────────────────── Stream A: histopathology ─────────────────────────────
STREAM_A = None
STREAM_A_ERROR = None
try:
    import torch
    from PIL import Image
    from models import IMAGENET_MEAN, IMAGENET_STD, SUBTYPE_NAMES, SUBTYPES, GradCAM, load_stream_a

    if WEIGHTS_PATH.exists():
        STREAM_A = load_stream_a(str(WEIGHTS_PATH))
    else:
        STREAM_A_ERROR = (f"Checkpoint not found at {WEIGHTS_PATH}. "
                          "Copy best_convnext_single_cbam.pth there (see weights/README.md).")
except Exception as e:
    STREAM_A_ERROR = f"Histopathology model not available: {e}"


def predict_histology(img):
    if STREAM_A is None:
        raise gr.Error(STREAM_A_ERROR)
    if img is None:
        raise gr.Error("Upload an H&E patch first.")
    rgb = np.array(Image.fromarray(img).convert("RGB").resize((224, 224), Image.BILINEAR))
    x = (rgb.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    tensor = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).requires_grad_(True)

    with torch.no_grad():
        probs = torch.softmax(STREAM_A(tensor), dim=1)[0].numpy()
    top = int(probs.argmax())

    cam_tool = GradCAM(STREAM_A, STREAM_A.backbone[-1])
    with torch.enable_grad():
        cam = cam_tool.generate(tensor, top)
    cam_tool.remove()
    cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
    heat = (plt.get_cmap("jet")(cam)[..., :3] * 255).astype(np.uint8)
    overlay = (0.55 * rgb + 0.45 * heat).astype(np.uint8)

    labels = {f"{s} - {SUBTYPE_NAMES[s]}": float(p) for s, p in zip(SUBTYPES, probs)}
    return labels, overlay


# ───────────────────────────── UI ─────────────────────────────
DISCLAIMER = ("**Research demo only** - built for a graduation project. "
              "It is not a medical device and must not be used for diagnosis.")

with gr.Blocks(title="Ovarian Cancer Dual-Stream AI") as demo:
    gr.Markdown("# Ovarian Cancer Dual-Stream AI\n"
                "Interpretable deep learning for ovarian cancer: **histopathology subtype classification** "
                "(ConvNeXt + CBAM + Grad-CAM) and **clinical malignancy prediction** (CatBoost + SHAP).\n\n"
                + DISCLAIMER)

    with gr.Tab("Clinical blood panel (Stream B)"):
        if CLINICAL is not None:
            m = CLINICAL.metrics
            gr.Markdown(f"Demo model retrained at start-up on 16 biomarkers. Held-out test set (n={m['n']}): "
                        f"accuracy {m['acc']:.1f}%, recall {m['recall']:.1f}%, ROC-AUC {m['auc']:.1f}%. "
                        "Fields start at the training-set median.")
        else:
            gr.Markdown(f"⚠️ {CLINICAL_ERROR}")
        inputs = []
        with gr.Row():
            for chunk in (FEATURES[:8], FEATURES[8:]):
                with gr.Column():
                    for f in chunk:
                        default = float(CLINICAL.medians[f]) if CLINICAL is not None else None
                        inputs.append(gr.Number(label=FEATURE_LABELS[f], value=default))
        btn_b = gr.Button("Predict", variant="primary")
        with gr.Row():
            out_label_b = gr.Label(label="Prediction", num_top_classes=2)
            out_plot_b = gr.Plot(label="SHAP explanation")
        btn_b.click(predict_clinical, inputs=inputs, outputs=[out_label_b, out_plot_b])

    with gr.Tab("Histopathology patch (Stream A)"):
        if STREAM_A is None:
            gr.Markdown(f"⚠️ {STREAM_A_ERROR}")
        gr.Markdown("Upload an H&E tissue patch. The model predicts one of 5 subtypes "
                    "(HGSC, EC, CC, MC, LGSC) and shows where it looked (Grad-CAM).")
        with gr.Row():
            in_img = gr.Image(label="H&E patch", type="numpy")
            out_cam = gr.Image(label="Grad-CAM overlay")
        btn_a = gr.Button("Classify", variant="primary")
        out_label_a = gr.Label(label="Subtype probabilities", num_top_classes=5)
        btn_a.click(predict_histology, inputs=in_img, outputs=[out_label_a, out_cam])

if __name__ == "__main__":
    demo.launch()
