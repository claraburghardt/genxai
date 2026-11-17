import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

from modules.data_ingestion import IngestionOutput
from modules.classification import ClassificationOutput


N_RUNS        = 10
N_TOP_OVERLAP = 20   # top-k genes used for overlap metric
RHO_THRESHOLD = 0.85  # minimum mean Spearman correlation to pass
OVERLAP_THRESHOLD = 0.80  # minimum mean top-k overlap to pass


@dataclass
class StabilityOutput:
    per_run_top_genes: dict        # seed -> subtype -> list of (gene, mean_abs_shap)
    spearman_matrix: dict          # subtype -> (n_runs x n_runs) correlation matrix
    overlap_matrix: dict           # subtype -> (n_runs x n_runs) overlap matrix
    mean_spearman: dict            # subtype -> mean rho across all run pairs
    mean_overlap: dict             # subtype -> mean overlap across all run pairs
    passed: dict                   # subtype -> bool
    summary: dict                  # overall pass/fail + aggregate stats


class StabilityModule:
    """
    Experiment 3 — SHAP explanation stability across retraining runs.

    Motivated by Oka and Takefuji (2026): feature importance from tree-based
    models can change across runs when the data is resampled. To test this for
    SHAP explanations, each of the N_RUNS runs repeats the WHOLE pipeline with a
    different random seed:
      1. New stratified train/test split (seed = run).
      2. New feature selection (variance + mutual information) on that run's training set.
      3. New SMOTE resampling on that run's training set.
      4. Retrain XGBoost with the best hyperparameters from Experiment 1.
      5. Compute TreeSHAP on that run's test set and rank genes by mean |SHAP| per subtype.

    Because the genes selected differ between runs, rankings are compared on the
    union of the top genes, and overlap is measured on gene identity (not index).

    Stability metrics across all run pairs (i, j):
      - Spearman rank correlation of mean |SHAP| over the shared genes  (target >= 0.85)
      - Top-20 gene overlap: |intersection| / N_TOP_OVERLAP            (target >= 0.80)

    A subtype passes if BOTH mean metrics meet their thresholds.
    """

    def __init__(self, config: dict):
        self.config = config["classification"]
        self.fs_config = config["feature_selection"]
        self.prep_config = config["preprocessing"]
        self.output_config = config["output"]
        self.stab_config = config.get("stability", {})
        self.n_runs = self.stab_config.get("n_runs", N_RUNS)

    def run(self, ingestion: IngestionOutput,
            classification: ClassificationOutput) -> StabilityOutput:

        mode = self.stab_config.get("mode", "refit")
        scenarios = ["fixed", "refit"] if mode == "both" else [mode]

        last_output = None
        for scenario in scenarios:
            last_output = self._run_scenario(scenario, ingestion, classification)

        # when running both, the returned object is the last scenario (refit)
        return last_output

    def _run_scenario(self, scenario: str,
                      ingestion: IngestionOutput,
                      classification: ClassificationOutput) -> StabilityOutput:

        best_params   = classification.cv_results.get("best_params", {})
        le            = classification.label_encoder
        classes       = list(le.classes_)
        fixed_genes   = classification.selected_genes

        X_full = ingestion.X.copy()
        if X_full.isnull().any().any():
            X_full = X_full.fillna(X_full.median())
        y_full = ingestion.y

        label = "A (fixed gene panel)" if scenario == "fixed" else "B (re-selected genes)"
        print(f"\n[Stability] Scenario {label} — {self.n_runs} runs ...")

        seeds = list(range(self.n_runs))
        per_run_top_genes = {}   # seed -> subtype -> list of (gene, mean_abs_shap)

        for seed in seeds:
            print(f"[Stability]   run {seed+1}/{self.n_runs} (seed={seed}) ...", end=" ", flush=True)

            # 1. new split with this run's seed
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_full, y_full,
                test_size=self.prep_config["test_size"],
                stratify=y_full,
                random_state=seed,
            )

            # 2. features: fixed panel (scenario A) or re-selected per run (scenario B)
            if scenario == "fixed":
                genes_run = [g for g in fixed_genes if g in X_tr.columns]
            else:
                genes_run = self._select_features(X_tr, y_tr, seed)
            X_tr_sel = X_tr[genes_run]
            X_te_sel = X_te[genes_run]

            # 3. SMOTE on this run's training set only
            if self.prep_config.get("apply_smote", True):
                sm = SMOTE(random_state=seed)
                X_tr_res, y_tr_res = sm.fit_resample(X_tr_sel, y_tr)
            else:
                X_tr_res, y_tr_res = X_tr_sel, y_tr

            # 4. retrain XGBoost with Experiment 1 hyperparameters
            model = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=-1,
                **best_params,
            )
            model.fit(X_tr_res, le.transform(y_tr_res))

            # 5. TreeSHAP on this run's test set
            explainer   = shap.TreeExplainer(model)
            shap_values = np.array(explainer.shap_values(X_te_sel))
            assert shap_values.ndim == 3 and shap_values.shape[-1] == len(classes), (
                f"Unexpected SHAP output shape {shap_values.shape}; expected "
                f"(n_samples, n_genes, {len(classes)}). Check the shap version (>=0.45)."
            )

            y_pred_enc = model.predict(X_te_sel)
            y_pred     = le.inverse_transform(y_pred_enc)

            run_top = {}
            for cls in classes:
                cls_idx = classes.index(cls)
                mask    = (y_te.values == cls) & (y_pred == cls)
                if mask.sum() == 0:
                    run_top[cls] = []
                    continue
                mean_abs = np.abs(shap_values[mask, :, cls_idx]).mean(axis=0)
                ranked   = np.argsort(mean_abs)[::-1]
                run_top[cls] = [(genes_run[i], float(mean_abs[i])) for i in ranked]

            per_run_top_genes[seed] = run_top
            print("done")

        # --- Compute pairwise metrics ---
        (spearman_matrix, overlap_matrix,
         mean_spearman, mean_overlap, passed) = self._compute_pairwise(classes, per_run_top_genes)

        n_passed = sum(passed.values())
        summary  = {
            "scenario":         scenario,
            "n_runs":           self.n_runs,
            "n_top_overlap":    N_TOP_OVERLAP,
            "rho_threshold":    RHO_THRESHOLD,
            "overlap_threshold": OVERLAP_THRESHOLD,
            "subtypes_passed":  n_passed,
            "subtypes_total":   len(classes),
            "mean_spearman_all": float(np.mean(list(mean_spearman.values()))),
            "mean_overlap_all":  float(np.mean(list(mean_overlap.values()))),
        }

        self._print_results(classes, mean_spearman, mean_overlap, passed, summary)
        self._save_results(classes, mean_spearman, mean_overlap, passed, summary, scenario)

        return StabilityOutput(
            per_run_top_genes=per_run_top_genes,
            spearman_matrix=spearman_matrix,
            overlap_matrix=overlap_matrix,
            mean_spearman=mean_spearman,
            mean_overlap=mean_overlap,
            passed=passed,
            summary=summary,
        )

    def _compute_pairwise(self, classes, per_run_top_genes):
        spearman_matrix = {}
        overlap_matrix  = {}
        mean_spearman   = {}
        mean_overlap    = {}
        passed          = {}

        for cls in classes:
            n = self.n_runs
            rho_mat     = np.ones((n, n))
            overlap_mat = np.ones((n, n))

            for i in range(n):
                rank_i = {g: s for g, s in per_run_top_genes[i][cls]}
                names_i = set(list(rank_i.keys())[:N_TOP_OVERLAP])

                for j in range(i + 1, n):
                    rank_j = {g: s for g, s in per_run_top_genes[j][cls]}
                    names_j = set(list(rank_j.keys())[:N_TOP_OVERLAP])

                    shared = [g for g in rank_i if g in rank_j]
                    if len(shared) > 1:
                        scores_i = [rank_i[g] for g in shared]
                        scores_j = [rank_j[g] for g in shared]
                        rho, _ = spearmanr(scores_i, scores_j)
                        if np.isnan(rho):
                            rho = 0.0
                    else:
                        rho = 0.0

                    overlap = len(names_i & names_j) / N_TOP_OVERLAP if names_i else 0.0

                    rho_mat[i, j] = rho_mat[j, i] = rho
                    overlap_mat[i, j] = overlap_mat[j, i] = overlap

            pairs_rho     = rho_mat[np.triu_indices(n, k=1)]
            pairs_overlap = overlap_mat[np.triu_indices(n, k=1)]

            spearman_matrix[cls] = rho_mat.tolist()
            overlap_matrix[cls]  = overlap_mat.tolist()
            mean_spearman[cls]   = float(np.mean(pairs_rho))
            mean_overlap[cls]    = float(np.mean(pairs_overlap))
            passed[cls]          = (mean_spearman[cls] >= RHO_THRESHOLD and
                                    mean_overlap[cls]  >= OVERLAP_THRESHOLD)

        return spearman_matrix, overlap_matrix, mean_spearman, mean_overlap, passed

    def _select_features(self, X_tr, y_tr, seed):
        """Variance filter + mutual-information top-N on this run's training set only."""
        # variance filter
        vt = VarianceThreshold(threshold=self.fs_config["variance_threshold"])
        vt.fit(X_tr)
        kept = X_tr.columns[vt.get_support()]
        X_v = X_tr[kept]

        # mutual information ranking
        le = LabelEncoder()
        y_enc = le.fit_transform(y_tr)
        mi = mutual_info_classif(X_v, y_enc, random_state=seed)
        n_top = min(self.fs_config["n_top_genes"], X_v.shape[1])
        top_idx = mi.argsort()[::-1][:n_top]
        return X_v.columns[top_idx].tolist()

    def _print_results(self, classes, mean_spearman, mean_overlap, passed, summary):
        label = "A — FIXED gene panel" if summary["scenario"] == "fixed" else "B — RE-SELECTED genes"
        print("\n" + "=" * 60)
        print(f"EXPERIMENT 3 — SHAP STABILITY — Scenario {label}")
        print("=" * 60)
        for cls in classes:
            status = "PASS" if passed[cls] else "FAIL"
            print(f"\n{cls}")
            print(f"  Mean Spearman rho : {mean_spearman[cls]:.4f}  (target ≥ {RHO_THRESHOLD})")
            print(f"  Mean top-{N_TOP_OVERLAP} overlap : {mean_overlap[cls]:.4f}  (target ≥ {OVERLAP_THRESHOLD})")
            print(f"  Result            : {status}")
        print("\n" + "=" * 60)
        print(f"Scenario        : {summary['scenario']}")
        print(f"Subtypes passed : {summary['subtypes_passed']}/{summary['subtypes_total']}")
        print(f"Mean rho (all)  : {summary['mean_spearman_all']:.4f}")
        print(f"Mean overlap    : {summary['mean_overlap_all']:.4f}")
        print("=" * 60)

    def _save_results(self, classes, mean_spearman, mean_overlap, passed, summary, scenario):
        results = {
            "experiment": 3,
            "summary": summary,
            "per_subtype": {
                cls: {
                    "mean_spearman": mean_spearman[cls],
                    "mean_overlap":  mean_overlap[cls],
                    "passed":        passed[cls],
                }
                for cls in classes
            },
        }
        path = self.output_config["results_path"].replace(".json", f"_exp3_{scenario}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Stability] Results saved to {path}")
