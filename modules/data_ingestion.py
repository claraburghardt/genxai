import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class IngestionOutput:
    X: pd.DataFrame          # samples x genes expression matrix
    y: pd.Series             # PAM50 subtype labels aligned with X
    gene_ids: list[str]      # gene identifiers (column names of X)
    sample_ids: list[str]    # sample identifiers (index of X)


class DataIngestionModule:
    """
    Loads RNA-seq gene expression matrix and PAM50 labels.
    Applies log2 normalization if raw counts are provided.
    """

    def __init__(self, config: dict):
        self.config = config["data"]

    def run(self) -> IngestionOutput:
        X = self._load_expression()
        y = self._load_labels(X.index)
        # align: keep only samples present in both X and y
        common = X.index.intersection(y.index)
        X, y = X.loc[common], y.loc[common]

        if self.config.get("primary_tumor_only", True):
            codes = pd.Series([s[13:15] for s in X.index], index=X.index)
            keep = codes == "01"
            dropped = (~keep).sum()
            X, y = X.loc[keep], y.loc[keep]
            print(f"[DataIngestion] Primary-tumor filter: dropped {dropped} non-primary samples (codes != 01)")

        # optionally drop the Normal-like class entirely (4-subtype setting, as in Park & Rhee 2024)
        if self.config.get("drop_normal_like", False):
            before = len(y)
            mask = y != "Normal"
            X, y = X.loc[mask], y.loc[mask]
            print(f"[DataIngestion] Dropped Normal-like class: {before} -> {len(y)} samples")

        print(f"[DataIngestion] Loaded {X.shape[0]} samples x {X.shape[1]} genes")
        print(f"[DataIngestion] PAM50 distribution:\n{y.value_counts()}\n")
        return IngestionOutput(
            X=X,
            y=y,
            gene_ids=X.columns.tolist(),
            sample_ids=X.index.tolist(),
        )

    def _load_expression(self) -> pd.DataFrame:
        path = self.config["expression_file"]
        df = pd.read_csv(path, sep="\t", index_col=0)

        # transpose so rows = samples, columns = genes
        if self.config.get("gene_axis", "columns") == "rows":
            df = df.T

        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)

        if not self.config.get("already_log2_normalized", True):
            df = np.log2(df + 1)
            print("[DataIngestion] Applied log2(x+1) normalization")

        return df

    def _load_labels(self, sample_index: pd.Index) -> pd.Series:
        path = self.config["labels_file"]
        column = self.config.get("labels_column", "PAM50Call_RNAseq")
        df = pd.read_csv(path, sep="\t", index_col=0)
        labels = df[column].dropna()
        labels.index = labels.index.astype(str)
        labels.name = "PAM50"
        return labels