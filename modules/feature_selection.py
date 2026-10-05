import pandas as pd
from dataclasses import dataclass
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from modules.preprocessing import PreprocessingOutput


@dataclass
class FeatureSelectionOutput:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    selected_genes: list[str]    # gene identifiers kept after selection


class FeatureSelectionModule:
    """
    Reduces the gene feature in two steps:
      1. Variance threshold: removes near-constant genes.
      2. Mutual information ranking: keeps the top N genes by discriminative power across PAM50 classes.
    Returns selected gene identifiers alongside the reduced matrices so downstream modules can label SHAP attributions.
    """

    def __init__(self, config: dict):
        self.config = config["feature_selection"]

    def run(self, preprocessed: PreprocessingOutput) -> FeatureSelectionOutput:
        X_train = preprocessed.X_train.copy()
        X_test = preprocessed.X_test.copy()
        y_train = preprocessed.y_train.copy()

        # step 1: variance filter
        X_train, X_test = self._variance_filter(X_train, X_test)

        # step 2: mutual information ranking
        X_train, X_test, selected_genes = self._mutual_info_filter(X_train, X_test, y_train)

        print(f"[FeatureSelection] Final feature set: {len(selected_genes)} genes\n")

        return FeatureSelectionOutput(
            X_train=X_train,
            X_test=X_test,
            y_train=preprocessed.y_train,
            y_test=preprocessed.y_test,
            selected_genes=selected_genes,
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
