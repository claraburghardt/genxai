import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from modules.data_ingestion import IngestionOutput


@dataclass
class PreprocessingOutput:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    gene_ids: list[str]
    sample_ids_train: list[str]
    sample_ids_test: list[str]
    X_train_raw: pd.DataFrame = None
    y_train_raw: pd.Series = None


class PreprocessingModule:
    """
    Runs quality filtering, variance-based gene removal, stratified train/test split, and SMOTE resampling on the training partition to address class imbalance.
    """

    def __init__(self, config: dict):
        self.config = config["preprocessing"]

    def run(self, ingestion: IngestionOutput) -> PreprocessingOutput:
        X, y = ingestion.X.copy(), ingestion.y.copy()

        X = self._filter_samples(X)
        y = y.loc[X.index]

        # fill remaining NaN with gene median (after sample filter, before gene filter)
        if X.isnull().any().any():
            X = X.fillna(X.median())

        X = self._filter_genes(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["test_size"],
            stratify=y,
            random_state=self.config["random_state"],
        )

        X_train_raw = X_train.copy()
        y_train_raw = y_train.copy()

        print(f"[Preprocessing] Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
        print(f"[Preprocessing] Genes after filtering: {X_train.shape[1]}\n")

        return PreprocessingOutput(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            gene_ids=X_train.columns.tolist(),
            sample_ids_train=X_train.index.tolist(),
            sample_ids_test=X_test.index.tolist(),
            X_train_raw=X_train_raw,
            y_train_raw=y_train_raw,
        )

    def _filter_samples(self, X: pd.DataFrame) -> pd.DataFrame:
        threshold = self.config["missing_value_threshold"]
        missing_rate = X.isnull().mean(axis=1)
        before = len(X)
        X = X[missing_rate <= threshold]
        print(f"[Preprocessing] Sample filter: {before} -> {len(X)} samples (dropped {before - len(X)})")
        return X

    def _filter_genes(self, X: pd.DataFrame) -> pd.DataFrame:
        min_frac = self.config["min_sample_expression"]
        if self.config.get("gene_filter_nonzero", True):
            expressed_frac = (X != 0).mean(axis=0)
        else:
            expressed_frac = X.notna().mean(axis=0)
        before = X.shape[1]
        X = X.loc[:, expressed_frac >= min_frac]
        print(f"[Preprocessing] Gene filter: {before} -> {X.shape[1]} genes (dropped {before - X.shape[1]})")
        return X
