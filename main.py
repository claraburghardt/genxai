import json
import yaml
import os
import argparse
from modules.data_ingestion import DataIngestionModule
from modules.preprocessing import PreprocessingModule
from modules.feature_selection import FeatureSelectionModule
from modules.classification import ClassificationModule


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Ingest → preprocess → feature selection → train/evaluate (Experiment 1)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true",
                        help="Load saved model instead of retraining (requires outputs/xgboost_model.json)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML file (default: config.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs("outputs", exist_ok=True)

    ingestion      = DataIngestionModule(config).run()
    preprocessed   = PreprocessingModule(config).run(ingestion)
    features       = FeatureSelectionModule(config).run(preprocessed)
    classification = ClassificationModule(config).run(features, skip_training=args.skip_training)

    # save selected genes for ClinicalInterface (app.py)
    import re
    genes_path = re.sub(r'results(_\w+)?\.json', r'selected_genes\1.json',
                        config["output"]["results_path"])
    with open(genes_path, "w") as f:
        json.dump(classification.selected_genes, f)

    print("\n[main] Pipeline finished. Model and selected genes saved to outputs/.")
    print("[main] Run 'python run_experiments.py' to reproduce Experiments 2 and 3.")


if __name__ == "__main__":
    main()
