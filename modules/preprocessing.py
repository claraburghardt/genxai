import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from modules.data_ingestion import IngestionOutput


@dataclass
class PreprocessingOutput:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    gene_ids: list[str]      # genes remaining after filtering
    sample_ids_train: list[str]
    sample_ids_test: list[str]


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

        X = self._filter_genes(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["test_size"],
            stratify=y,
            random_state=self.config["random_state"],
        )

        if self.config.get("apply_smote", True):
            X_train, y_train = self._apply_smote(X_train, y_train)

        print(f"[Preprocessing] Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
        print(f"[Preprocessing] Genes after filtering: {X_train.shape[1]}")
        print(f"[Preprocessing] Train PAM50 distribution after SMOTE:\n{y_train.value_counts()}\n")

        return PreprocessingOutput(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            gene_ids=X_train.columns.tolist(),
            sample_ids_train=X_train.index.tolist(),
            sample_ids_test=X_test.index.tolist(),
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
        # keep genes expressed (non-zero) in at least min_frac of samples
        expressed_frac = (X > 0).mean(axis=0)
        before = X.shape[1]
        X = X.loc[:, expressed_frac >= min_frac]
        print(f"[Preprocessing] Gene filter: {before} -> {X.shape[1]} genes (dropped {before - X.shape[1]})")
        return X

    def _apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series):
        smote = SMOTE(random_state=self.config["random_state"])
        X_res, y_res = smote.fit_resample(X_train, y_train)
        # restore DataFrame/Series with gene names
        X_res = pd.DataFrame(X_res, columns=X_train.columns)
        y_res = pd.Series(y_res, name=y_train.name)
        print(f"[Preprocessing] SMOTE: {len(y_train)} -> {len(y_res)} training samples")
        return X_res, y_res
