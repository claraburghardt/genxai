import pandas as pd
from dataclasses import dataclass
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from modules.preprocessing import PreprocessingOutput


@dataclass
class FeatureSelectionOutput:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    selected_genes: list[str]
    X_train_raw: pd.DataFrame = None
    y_train_raw: pd.Series = None


class FeatureSelectionModule:
    """
    Reduces the gene feature space in two steps, then balances the classes:
      1. Variance threshold: removes near-constant genes.
      2. Mutual information ranking: keeps the top N genes by discriminative power across PAM50 classes.
      3. SMOTE: oversamples the minority classes AFTER selection.

    Feature selection is run on the REAL training samples only (before SMOTE),
    so the selected genes are not driven by synthetic samples. SMOTE is applied
    afterwards, on the reduced gene set, to balance the classes for training.
    Returns selected gene identifiers alongside the reduced matrices so
    downstream modules can label SHAP attributions.
    """

    def __init__(self, config: dict):
        self.config = config["feature_selection"]
        self.prep_config = config["preprocessing"]

    def run(self, preprocessed: PreprocessingOutput) -> FeatureSelectionOutput:
        X_train = preprocessed.X_train_raw.copy()
        y_train = preprocessed.y_train_raw.copy()
        X_test = preprocessed.X_test.copy()

        # step 1: variance filter
        X_train, X_test = self._variance_filter(X_train, X_test)

        # step 2: mutual information
        X_train, X_test, selected_genes = self._mutual_info_filter(X_train, X_test, y_train)

        # step 3: SMOTE
        if self.prep_config.get("apply_smote", True):
            X_train, y_train = self._apply_smote(X_train, y_train, selected_genes)

        print(f"[FeatureSelection] Final feature set: {len(selected_genes)} genes\n")

        return FeatureSelectionOutput(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=preprocessed.y_test,
            selected_genes=selected_genes,
            X_train_raw=preprocessed.X_train_raw,
            y_train_raw=preprocessed.y_train_raw,
        )

    def _variance_filter(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        threshold = self.config["variance_threshold"]
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_train)
        mask = selector.get_support()
        before = X_train.shape[1]
        X_train = X_train.loc[:, mask]
        X_test = X_test.loc[:, mask]
        print(f"[FeatureSelection] Variance filter: {before} -> {X_train.shape[1]} genes")
        return X_train, X_test

    def _mutual_info_filter(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series):
        n_top = self.config["n_top_genes"]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train)
        mi_scores = mutual_info_classif(X_train, y_encoded, random_state=42)
        top_indices = mi_scores.argsort()[::-1][:n_top]
        selected_genes = X_train.columns[top_indices].tolist()
        before = X_train.shape[1]
        X_train = X_train[selected_genes]
        X_test = X_test[selected_genes]
        print(f"[FeatureSelection] Mutual information filter: {before} -> {len(selected_genes)} genes")
        return X_train, X_test, selected_genes

    def _apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series, selected_genes: list):
        smote = SMOTE(random_state=self.prep_config["random_state"])
        X_res, y_res = smote.fit_resample(X_train, y_train)
        X_res = pd.DataFrame(X_res, columns=selected_genes)
        y_res = pd.Series(y_res, name=y_train.name)
        print(f"[FeatureSelection] SMOTE: {len(y_train)} -> {len(y_res)} training samples (after selection)")
        return X_res, y_res
