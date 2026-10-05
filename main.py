import yaml
import os
from modules.data_ingestion import DataIngestionModule
from modules.preprocessing import PreprocessingModule
from modules.feature_selection import FeatureSelectionModule


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    os.makedirs("outputs", exist_ok=True)

    # --- Week 1: Data pipeline ---
    ingestion = DataIngestionModule(config).run()
    preprocessed = PreprocessingModule(config).run(ingestion)
    features = FeatureSelectionModule(config).run(preprocessed)

    print("Pipeline complete.")
    print(f"  Training set : {features.X_train.shape}")
    print(f"  Test set     : {features.X_test.shape}")
    print(f"  Selected genes (first 10): {features.selected_genes[:10]}")


if __name__ == "__main__":
    main()
