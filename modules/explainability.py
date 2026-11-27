import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import shap
import gseapy as gp

from modules.classification import ClassificationOutput

# PAM50 gene panel (Parker et al. 2009) — identical for all subtypes
_PAM50_GENES = {
    "ACTR3B","ANLN","BAG1","BCL2","BIRC5","BLVRA","CCNB1","CCNE1",
    "CDC20","CDC6","CDCA1","CDH3","CENPF","CEP55","CXXC5","EGFR",
    "ERBB2","ESR1","EXO1","FGFR4","FOXA1","FOXC1","GPR160","GRB7",
    "KIF2C","KRT14","KRT17","KRT5","LAPTM4B","MAPT","MDM2","MELK",
    "MIA","MKI67","MLPH","MMP11","MYBL2","MYC","NAT1","NDC80",
    "NUF2","ORC6L","PGR","PHGDH","PTTG1","RRM2","SFRP1","SLC39A6",
    "TMEM45B","TYMS","UBE2C","UBE2T",
}

# Subtype-specific literature markers (Parker et al. 2009 + established biomarkers)
PAM50_PANEL = {
    "Basal":  {"pam50": _PAM50_GENES, "literature": {"EGFR","KRT5","KRT17","KRT14","CDH3","FOXC1","PHGDH"}},
    "Her2":   {"pam50": _PAM50_GENES, "literature": {"ERBB2","GRB7","FGFR4","CXXC5","GPR160"}},
    "LumA":   {"pam50": _PAM50_GENES, "literature": {"ESR1","FOXA1","GATA3","PGR","BCL2","NAT1","MAPT","SLC39A6"}},
    "LumB":   {"pam50": _PAM50_GENES, "literature": {"MKI67","CCNB1","MYBL2","MELK","CDC20","BIRC5","RRM2"}},
    "Normal": {"pam50": _PAM50_GENES, "literature": {"ESR1","PGR","FOXA1","BCL2","MIA","MLPH"}},
}

BREAST_CANCER_PATHWAYS = [
    "estrogen response", "estrogen signaling", "androgen response",
    "endocrine resistance", "breast cancer", "erbb", "egfr",
    "cell cycle", "g2m checkpoint", "e2f targets", "mitotic spindle",
    "myc targets", "mtorc1", "pi3k", "p53", "mapk",
    "epithelial mesenchymal transition", "ecm", "focal adhesion",
    "hypoxia", "angiogenesis", "apoptosis", "pathways in cancer",
    "proteoglycans in cancer", "prolactin",
]

ENRICHMENT_FDR         = 0.05
N_TOP_COHERENCE        = 10
N_TOP_PATHWAY          = 100
MIN_CORRECT_SAMPLES    = 3   # subtypes with fewer correct predictions → "inconclusive"


@dataclass
class ExplainabilityOutput:
    shap_values: np.ndarray
    top_genes_per_subtype: dict
    coherence_results: dict
    enrichment_results: dict
    enrichment_summary: dict


class ExplainabilityModule:
    """
    Experiment 2 — SHAP explanation quality and biological coherence.

    Steps:
      1. TreeSHAP: compute per-sample, per-gene SHAP values for all test samples.
      2. Rank genes per subtype by mean absolute SHAP across correctly classified
         samples (top-100 kept for enrichment; top-10 reported for illustration).
      3. Coherence check: the top genes of a subtype must be enriched for at least
         one breast-cancer-relevant pathway (FDR < 0.05). The overlap of the top-10
         with the PAM50 panel / markers is reported for illustration only.
      4. Pathway enrichment: ORA on the top-100 SHAP genes per subtype using
         MSigDB Hallmark 2020 and KEGG 2021 Human, FDR < 0.05.
    """

    def __init__(self, config: dict):
        self.output_config = config["output"]

    def run(self, classification: ClassificationOutput) -> ExplainabilityOutput:
        model   = classification.model
        le      = classification.label_encoder
        genes   = classification.selected_genes
        y_test  = classification.y_test
        y_pred  = classification.y_pred

        X_test_df = classification.X_test_df

        print("[Explainability] Computing TreeSHAP values ...")
        explainer   = shap.TreeExplainer(model)
        shap_values = np.array(explainer.shap_values(X_test_df))
        assert shap_values.ndim == 3 and shap_values.shape[-1] == len(le.classes_), (
            f"Unexpected SHAP output shape {shap_values.shape}; expected "
            f"(n_samples, n_genes, {len(le.classes_)}). Check the shap version (>=0.45)."
        )
        print(f"[Explainability] SHAP matrix shape: {shap_values.shape}")

        classes = list(le.classes_)

        # --- Top genes per subtype ---
        top_genes = self._top_genes_per_subtype(shap_values, X_test_df, y_test, y_pred, genes, classes)

        n_correct = {
            c: int(((y_test.values == c) & (y_pred == c)).sum())
            for c in classes
        }
        print("[Explainability] Correct predictions per subtype:", n_correct)

        # --- Coherence check ---
        coherence = self._check_coherence(top_genes, classes)

        # --- Pathway enrichment ---
        enrichment, enrich_summary = self._pathway_enrichment(top_genes, classes, n_correct)

        self._print_results(top_genes, coherence, enrich_summary, classes)
        self._save_results(top_genes, coherence, enrich_summary)

        return ExplainabilityOutput(
            shap_values=np.array(shap_values),
            top_genes_per_subtype=top_genes,
            coherence_results=coherence,
            enrichment_results=enrichment,
            enrichment_summary=enrich_summary,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def explain_sample(model, genes: list, X: np.ndarray, pred_class: str,
                       classes: list, n_top: int = 100):
        """
        Compute SHAP values and pathway enrichment for a single sample.
        Returns (shap_vals_1d, base_value, top_genes, enrich_df).
        Used by ClinicalInterface for on-the-fly inference.
        """
        explainer   = shap.TreeExplainer(model)
        shap_values = np.array(explainer.shap_values(X))  # (n_samples, n_genes, n_classes)
        cls_idx     = classes.index(pred_class)
        shap_sample = shap_values[0, :, cls_idx]
        base_value  = (explainer.expected_value[cls_idx]
                       if hasattr(explainer.expected_value, "__len__")
                       else float(explainer.expected_value))

        ranked     = np.argsort(np.abs(shap_sample))[::-1]
        top_genes  = [(genes[i], float(shap_sample[i])) for i in ranked[:20]]
        gene_list  = [genes[i] for i in ranked[:n_top]]

        enrich_error = None
        try:
            enr = gp.enrichr(
                gene_list  = gene_list,
                gene_sets  = ["MSigDB_Hallmark_2020", "KEGG_2021_Human"],
                organism   = "human",
                outdir     = None,
                cutoff     = 0.05,
                verbose    = False,
            )
            enrich_df = enr.results[enr.results["Adjusted P-value"] < 0.05].copy()
            enrich_df = enrich_df.sort_values("Adjusted P-value").head(10)
        except Exception as e:
            enrich_df = pd.DataFrame()
            enrich_error = str(e)

        return shap_sample, base_value, top_genes, enrich_df, enrich_error

    def _top_genes_per_subtype(self, shap_values, X_test_df, y_test, y_pred, genes, classes):
        shap_arr = np.array(shap_values)   # (n_samples, n_genes, n_classes)
        top_genes = {}

        for cls in classes:
            cls_idx = classes.index(cls)
            mask = (y_test.values == cls) & (y_pred == cls)
            if mask.sum() == 0:
                top_genes[cls] = []
                continue

            # mean absolute SHAP for this class across correct samples
            mean_abs = np.abs(shap_arr[mask, :, cls_idx]).mean(axis=0)
            ranked   = np.argsort(mean_abs)[::-1]

            top_genes[cls] = [
                (genes[i], float(mean_abs[i]))
                for i in ranked[:N_TOP_PATHWAY]
            ]

        return top_genes

    # Maps full subtype names (used by labels) to PAM50_PANEL keys
    _SUBTYPE_KEY = {
        "Basal":         "Basal",
        "HER2-enriched": "Her2",
        "Her2":          "Her2",
        "Luminal A":     "LumA",
        "LumA":          "LumA",
        "Luminal B":     "LumB",
        "LumB":          "LumB",
        "Normal":        "Normal",
        "Normal-like":   "Normal",
    }

    def _check_coherence(self, top_genes, classes):
        coherence = {}
        for cls in classes:
            genes_top10 = [g for g, _ in top_genes[cls][:N_TOP_COHERENCE]]
            panel_key   = self._SUBTYPE_KEY.get(cls, cls)
            reference   = PAM50_PANEL[panel_key]["pam50"] | PAM50_PANEL[panel_key]["literature"]
            hits        = [g for g in genes_top10 if g in reference]
            coherence[cls] = {
                "top10_genes": genes_top10,
                "hits":        hits,
                "n_hits":      len(hits),
            }
        return coherence

    def _pathway_enrichment(self, top_genes, classes, n_correct):
        enrichment  = {}
        summary     = {}

        for cls in classes:
            gene_list = [g for g, _ in top_genes[cls][:N_TOP_PATHWAY]]
            if not gene_list:
                enrichment[cls] = pd.DataFrame()
                summary[cls]    = {
                    "n_significant": 0, "top_pathway": None, "passed": False,
                    "n_correct": n_correct[cls], "status": "inconclusive",
                }
                continue

            try:
                enr = gp.enrichr(
                    gene_list   = gene_list,
                    gene_sets   = ["MSigDB_Hallmark_2020", "KEGG_2021_Human"],
                    organism    = "human",
                    outdir      = None,
                    cutoff      = ENRICHMENT_FDR,
                    verbose     = False,
                )
                results_df = enr.results[enr.results["Adjusted P-value"] < ENRICHMENT_FDR].copy()
                results_df = results_df.sort_values("Adjusted P-value")

                n_sig      = len(results_df)
                top_path   = results_df.iloc[0]["Term"] if n_sig > 0 else None

                passed     = any(
                    any(bc.lower() in term.lower() for bc in BREAST_CANCER_PATHWAYS)
                    for term in results_df["Term"].tolist()
                ) if n_sig > 0 else False

                nc = n_correct[cls]
                status = (
                    "inconclusive" if nc < MIN_CORRECT_SAMPLES
                    else "coherent" if passed else "not_coherent"
                )

                enrichment[cls] = results_df
                summary[cls]    = {
                    "n_significant": n_sig,
                    "top_pathway":   top_path,
                    "passed":        passed,
                    "n_correct":     nc,
                    "status":        status,
                }

            except Exception as e:
                print(f"[Explainability] Enrichment failed for {cls}: {e}")
                enrichment[cls] = pd.DataFrame()
                summary[cls]    = {
                    "n_significant": 0, "top_pathway": None, "passed": False,
                    "n_correct": n_correct[cls], "status": "inconclusive",
                }

        return enrichment, summary

    def _print_results(self, top_genes, coherence, enrich_summary, classes):
        print("\n" + "=" * 60)
        print("EXPERIMENT 2 — SHAP PATHWAY ENRICHMENT (biological coherence)")
        print("=" * 60)
        for cls in classes:
            c = coherence[cls]
            e = enrich_summary[cls]
            print(f"\n{cls}")
            print(f"  Top-10 SHAP genes  : {c['top10_genes']}")
            print(f"  PAM50/marker genes : {c['hits']} ({c['n_hits']}/{N_TOP_COHERENCE}) [illustrative]")
            print(f"  Correct samples    : {e['n_correct']}")
            print(f"  Enriched pathways  : {e['n_significant']} significant (FDR<{ENRICHMENT_FDR})")
            print(f"  Top pathway        : {e['top_pathway']}")
            print(f"  Status             : {e['status'].upper()}")
        print("\n" + "=" * 60)
        n_coh = sum(e["status"] == "coherent"      for e in enrich_summary.values())
        n_inc = sum(e["status"] == "inconclusive"  for e in enrich_summary.values())
        n_cls = len(enrich_summary)
        print(f"Biological coherence: {n_coh}/{n_cls} coherent, {n_inc} inconclusive")
        print("=" * 60)

    def _save_results(self, top_genes, coherence, enrich_summary):
        results = {
            "experiment": 2,
            "top_genes_per_subtype": {
                cls: [(g, s) for g, s in genes]
                for cls, genes in top_genes.items()
            },
            "coherence": {
                cls: {k: v for k, v in c.items()}
                for cls, c in coherence.items()
            },
            "enrichment_summary": enrich_summary,
        }
        path = self.output_config["results_path"].replace(".json", "_exp2.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Explainability] Results saved to {path}")
