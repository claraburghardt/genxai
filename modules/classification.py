import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from modules.feature_selection import FeatureSelectionOutput


def _mi_score(X, y):
    """Mutual information scorer with a fixed seed (picklable for n_jobs > 1)."""
    return mutual_info_classif(X, y, random_state=42)


@dataclass
class ClassificationOutput:
    model: XGBClassifier
    label_encoder: LabelEncoder
    selected_genes: list[str]
    cv_results: dict                  # per-fold metrics from cross-validation
    test_metrics: dict                # final metrics on the held-out test set
    X_test_df: pd.DataFrame           # feature matrix of test set (needed by ExplainabilityModule)
    y_test: pd.Series
    y_pred: np.ndarray
    y_prob: np.ndarray                # predicted probabilities, shape (n_test, n_classes)


class ClassificationModule:
    """
    Trains an XGBoost multi-class classifier for PAM50 subtype prediction.

    Steps:
      1. GridSearchCV over learning_rate, max_depth, n_estimators using
         stratified 5-fold CV on the training set.
      2. Refit the best estimator on the full training set.
      3. Evaluate on the held-out test set: accuracy, macro F1, AUC-OvR.
    """

    def __init__(self, config: dict):
        self.config = config["classification"]
        self.fs_config = config["feature_selection"]
        self.prep_config = config["preprocessing"]
        self.output_config = config["output"]

    def run(self, features: FeatureSelectionOutput, skip_training: bool = False) -> ClassificationOutput:
        X_train = features.X_train
        X_test = features.X_test
        y_train = features.y_train
        y_test = features.y_test

        # XGBoost requires integer class labels
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        print(f"[Classification] Classes: {list(le.classes_)}")

        if skip_training:
            print(f"[Classification] Loading saved model from {self.output_config['model_path']} ...")
            model = XGBClassifier()
            model.load_model(self.output_config["model_path"])
            cv_results = {"note": "skipped — loaded from disk"}
        else:
            print(f"[Classification] Training set: {X_train.shape[0]} samples x {X_train.shape[1]} genes")
            best_params, cv_results = self._grid_search(features.X_train_raw, features.y_train_raw, le)
            model = self._train_final_model(X_train, y_train_enc, best_params)

        # --- Evaluate on held-out test set ---
        y_pred_enc = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        y_pred = le.inverse_transform(y_pred_enc)
        test_metrics = self._compute_metrics(y_test_enc, y_pred_enc, y_prob, le)

        self._print_results(cv_results, test_metrics, y_test, y_pred, le)
        if not skip_training:
            self._save_results(cv_results, test_metrics)

        return ClassificationOutput(
            model=model,
            label_encoder=le,
            selected_genes=features.selected_genes,
            cv_results=cv_results,
            test_metrics=test_metrics,
            X_test_df=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grid_search(self, X_train_raw, y_train_raw, le):
        """Leakage-free cross-validation with feature selection and SMOTE inside each fold."""
        y_enc = le.transform(y_train_raw)

        param_grid = {
            f"clf__{k}": v for k, v in self.config["hyperparameter_grid"].items()
        }

        cv = StratifiedKFold(
            n_splits=self.config["cv_folds"],
            shuffle=True,
            random_state=self.config["random_state"],
        )

        apply_smote = self.prep_config.get("apply_smote", True)

        if apply_smote:
            pipeline = ImbPipeline(steps=[
                ("variance", VarianceThreshold(threshold=self.fs_config["variance_threshold"])),
                ("mi", SelectKBest(
                    score_func=_mi_score,
                    k=min(self.fs_config["n_top_genes"], X_train_raw.shape[1]),
                )),
                ("smote", SMOTE(random_state=self.config["random_state"])),
                ("clf", XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=self.config["random_state"],
                    n_jobs=-1,
                )),
            ])
        else:
            from sklearn.pipeline import Pipeline as SkPipeline
            pipeline = SkPipeline(steps=[
                ("variance", VarianceThreshold(threshold=self.fs_config["variance_threshold"])),
                ("mi", SelectKBest(
                    score_func=_mi_score,
                    k=min(self.fs_config["n_top_genes"], X_train_raw.shape[1]),
                )),
                ("clf", XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=self.config["random_state"],
                    n_jobs=-1,
                )),
            ])

        print(f"[Classification] Running leakage-free GridSearchCV ({self.config['cv_folds']}-fold) ...")
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=False,
        )
        gs.fit(X_train_raw, y_enc)

        best_params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        best_cv_score = gs.best_score_

        best_idx = gs.best_index_
        n_folds = self.config["cv_folds"]
        fold_scores = [
            gs.cv_results_[f"split{i}_test_score"][best_idx]
            for i in range(n_folds)
        ]

        cv_results = {
            "best_params": best_params,
            "best_cv_f1_macro": float(best_cv_score),
            "fold_f1_scores": [float(s) for s in fold_scores],
            "fold_f1_mean": float(np.mean(fold_scores)),
            "fold_f1_std": float(np.std(fold_scores)),
        }

        print(f"[Classification] Best params: {best_params}")
        print(f"[Classification] CV macro F1: {best_cv_score:.4f} ± {np.std(fold_scores):.4f}")
        return best_params, cv_results

    def _train_final_model(self, X_train, y_train_enc, best_params):
        model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=self.config["random_state"],
            n_jobs=-1,
            **best_params,
        )
        model.fit(X_train, y_train_enc)
        model.save_model(self.output_config["model_path"])
        print(f"[Classification] Model saved to {self.output_config['model_path']}")
        return model

    def _compute_metrics(self, y_test_enc, y_pred_enc, y_prob, le):
        n_classes = len(le.classes_)
        acc = accuracy_score(y_test_enc, y_pred_enc)
        f1_macro = f1_score(y_test_enc, y_pred_enc, average="macro")
        f1_per_class = f1_score(y_test_enc, y_pred_enc, average=None)

        auc_ovr = roc_auc_score(
            y_test_enc, y_prob,
            multi_class="ovr",
            average="macro",
        )

        return {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "auc_ovr_macro": float(auc_ovr),
            "f1_per_class": {
                le.classes_[i]: float(f1_per_class[i]) for i in range(n_classes)
            },
        }

    def _print_results(self, cv_results, test_metrics, y_test, y_pred, le):
        print("\n" + "=" * 60)
        print("EXPERIMENT 1 — TEST SET RESULTS")
        print("=" * 60)
        print(f"  Accuracy      : {test_metrics['accuracy']:.4f}")
        print(f"  Macro F1      : {test_metrics['f1_macro']:.4f}  (target ≥ 0.85)")
        print(f"  AUC-OvR macro : {test_metrics['auc_ovr_macro']:.4f}  (target ≥ 0.95)")
        print("\nF1 per subtype:")
        for subtype, score in test_metrics["f1_per_class"].items():
            print(f"  {subtype:<10} {score:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        print("Confusion matrix (rows = true, cols = predicted):")
        cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        print(cm_df.to_string())
        print("=" * 60)

    def _save_results(self, cv_results, test_metrics):
        results = {
            "experiment": 1,
            "cv_results": cv_results,
            "test_metrics": test_metrics,
        }
        with open(self.output_config["results_path"], "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Classification] Results saved to {self.output_config['results_path']}")
