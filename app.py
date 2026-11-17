import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from modules.explainability import ExplainabilityModule

MODEL_PATH   = "outputs/xgboost_model.json"
RESULTS_PATH = "outputs/results.json"
GENES_PATH   = "outputs/selected_genes.json"

SUBTYPE_COLORS = {
    "LumA":   "#4C72B0",
    "LumB":   "#55A868",
    "Basal":  "#C44E52",
    "Her2":   "#DD8452",
    "Normal": "#8172B3",
}

PAM50_CLASSES = ["Basal", "Her2", "LumA", "LumB", "Normal"]


class ClinicalInterface:
    """Streamlit clinical decision support interface for GenXAI."""

    def __init__(self):
        self.model = None
        self.le = None
        self.genes = None

    # ── loading ────────────────────────────────────────────────────────────

    @staticmethod
    @st.cache_resource
    def _load_model():
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        le = LabelEncoder()
        le.classes_ = np.array(PAM50_CLASSES)
        return model, le

    @staticmethod
    @st.cache_resource
    def _load_genes():
        with open(GENES_PATH) as f:
            return json.load(f)

    @staticmethod
    @st.cache_data
    def _load_results():
        with open(RESULTS_PATH) as f:
            return json.load(f)

    # ── plotting ─────────────────────────────────────────────────────────────

    @staticmethod
    def _shap_bar_fig(shap_vals: np.ndarray, genes: list[str],
                            predicted_class: str, n_show: int = 15) -> plt.Figure:
        ranked = np.argsort(np.abs(shap_vals))[::-1][:n_show]
        vals   = shap_vals[ranked]
        labels = [genes[i] for i in ranked]

        fig, ax = plt.subplots(figsize=(8, max(4, n_show * 0.35)))
        colors = ["#C44E52" if v > 0 else "#4C72B0" for v in vals]
        ax.barh(range(len(vals)), vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels[::-1], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value")
        ax.set_title(f"Top-{n_show} gene attributions — {predicted_class}", fontsize=11)
        fig.tight_layout()
        return fig

    # ── UI sections ──────────────────────────────────────────────────────────

    def render_sidebar(self):
        with st.sidebar:
            st.title("🧬 GenXAI")
            st.caption("PAM50 Breast Cancer Subtype Classifier")
            st.divider()

            uploaded = st.file_uploader(
                "Upload gene expression file",
                type=["tsv", "csv", "txt"],
                help="Tab or comma-separated: rows = genes, one sample column",
            )

            st.divider()
            st.markdown("**Model performance (test set)**")
            try:
                res = self._load_results()
                m   = res["test_metrics"]
                st.metric("Macro F1", f"{m['f1_macro']:.4f}")
                st.metric("AUC-OvR",  f"{m['auc_ovr_macro']:.4f}")
                st.metric("Accuracy", f"{m['accuracy']:.4f}")
            except Exception:
                st.info("Run the pipeline first to see metrics.")
        return uploaded

    def parse_upload(self, uploaded):
        sep = "\t" if uploaded.name.endswith((".tsv", ".txt")) else ","
        try:
            raw = pd.read_csv(uploaded, sep=sep, index_col=0)
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            st.stop()

        if raw.shape[0] > raw.shape[1]:
            raw = raw.T

        st.success(f"Loaded {raw.shape[0]} sample(s) × {raw.shape[1]} genes.")
        return raw

    def predict(self, raw):
        missing = [g for g in self.genes if g not in raw.columns]
        if len(missing) > len(self.genes) * 0.05:
            st.error(
                f"{len(missing)}/{len(self.genes)} required genes are missing "
                f"(more than 5%). The prediction would be unreliable. "
                "Make sure the file uses TCGA gene symbols and includes the selected genes."
            )
            st.stop()
        if missing:
            st.warning(f"{len(missing)} of {len(self.genes)} genes are missing and were filled with zero.")

        X = raw.reindex(columns=self.genes, fill_value=0.0)

        if float(np.nanmax(X.values)) > 100:
            st.warning(
                "The expression values look larger than expected for log2-normalized data. "
                "This model was trained on log2-normalized counts; raw counts may give a wrong prediction."
            )

        proba       = self.model.predict_proba(X)
        pred_enc    = np.argmax(proba, axis=1)
        pred_labels = self.le.inverse_transform(pred_enc)
        return X, proba, pred_labels

    def render_prediction(self, raw, X, proba, pred_labels):
        sample_idx = 0
        if X.shape[0] > 1:
            sample_idx = st.selectbox("Select sample", options=range(X.shape[0]),
                                      format_func=lambda i: raw.index[i])

        pred  = pred_labels[sample_idx]
        conf  = proba[sample_idx]
        color = SUBTYPE_COLORS.get(pred, "#888888")

        st.divider()
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Predicted subtype")
            st.markdown(
                f"<div style='background:{color};padding:16px;border-radius:10px;"
                f"text-align:center;color:white;font-size:2rem;font-weight:bold'>"
                f"{pred}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown("**Confidence scores**")
            conf_df = pd.DataFrame({
                "Subtype":     self.le.classes_,
                "Probability": conf,
            }).sort_values("Probability", ascending=False)
            st.dataframe(conf_df.style.format({"Probability": "{:.3f}"}), hide_index=True)

        with col2:
            st.markdown("### Confidence distribution")
            fig_conf, ax_conf = plt.subplots(figsize=(6, 3))
            bar_colors = [SUBTYPE_COLORS.get(c, "#888") for c in self.le.classes_]
            ax_conf.bar(self.le.classes_, conf, color=bar_colors)
            ax_conf.set_ylabel("Probability")
            ax_conf.set_ylim(0, 1)
            ax_conf.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
            fig_conf.tight_layout()
            st.pyplot(fig_conf, use_container_width=True)

        return X, pred, sample_idx

    def render_explanation(self, X, pred, sample_idx):
        st.divider()
        st.markdown("### Gene-level SHAP attributions")

        X_sample = X.iloc[[sample_idx]].values
        with st.spinner("Computing SHAP values and pathway enrichment ..."):
            shap_sample, base_val, top_genes, enr_df, enr_error = ExplainabilityModule.explain_sample(
                model      = self.model,
                genes      = self.genes,
                X          = X_sample,
                pred_class = pred,
                classes    = list(self.le.classes_),
            )

        fig_shap = self._shap_bar_fig(shap_sample, self.genes, pred)
        st.pyplot(fig_shap, use_container_width=True)

        top_genes_df = pd.DataFrame({
            "Gene":       [g for g, _ in top_genes],
            "SHAP value": [v for _, v in top_genes],
            "Direction":  ["↑ activates" if v > 0 else "↓ suppresses" for _, v in top_genes],
        })
        st.dataframe(top_genes_df.style.format({"SHAP value": "{:.4f}"}), hide_index=True)
        return enr_df, enr_error

    def render_pathways(self, enr_df, enr_error=None):
        st.divider()
        st.markdown("### Pathway enrichment (MSigDB Hallmark, FDR < 0.05)")

        if enr_error is not None:
            st.error(
                "Pathway enrichment could not be computed (the Enrichr request failed, "
                f"e.g. no internet connection). Details: {enr_error}"
            )
        elif enr_df.empty:
            st.info("No significantly enriched pathways found (FDR < 0.05) for this sample.")
        else:
            display_cols = ["Term", "Overlap", "Adjusted P-value", "Genes"]
            available    = [c for c in display_cols if c in enr_df.columns]
            st.dataframe(
                enr_df[available].style.format({"Adjusted P-value": "{:.2e}"}),
                hide_index=True,
                use_container_width=True,
            )

    # ── orchestration ────────────────────────────────────────────────────────

    def render(self):
        st.set_page_config(
            page_title="GenXAI — PAM50 Subtype Classifier",
            page_icon="🧬",
            layout="wide",
        )

        uploaded = self.render_sidebar()

        st.title("GenXAI — Interpretable PAM50 Subtype Classification")
        st.markdown(
            "Upload a gene expression profile to receive a PAM50 subtype prediction "
            "with SHAP gene-level attributions and pathway enrichment context."
        )

        if uploaded is None:
            st.info("👈 Upload a gene expression file in the sidebar to begin.")
            st.stop()

        self.model, self.le = self._load_model()
        self.genes          = self._load_genes()

        raw = self.parse_upload(uploaded)
        X, proba, pred_labels = self.predict(raw)
        X, pred, sample_idx   = self.render_prediction(raw, X, proba, pred_labels)
        enr_df, enr_error     = self.render_explanation(X, pred, sample_idx)
        self.render_pathways(enr_df, enr_error)

        st.divider()
        st.caption(
            "GenXAI · PAM50 subtype classification using XGBoost + TreeSHAP · "
            "TCGA-BRCA RNA-seq · Unisinos TCC 2026"
        )


ClinicalInterface().render()
