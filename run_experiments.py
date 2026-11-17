import json
import yaml
from modules.data_ingestion import DataIngestionModule
from modules.preprocessing import PreprocessingModule
from modules.feature_selection import FeatureSelectionModule
from modules.classification import ClassificationModule
from modules.explainability import ExplainabilityModule
from modules.stability import StabilityModule


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Experiments 2 (SHAP coherence) and 3 (stability). Requires main.py to have run first."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML file (default: config.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)

    best_params = {}
    try:
        with open(config["output"]["results_path"]) as f:
            saved = json.load(f)
        best_params = saved.get("cv_results", {}).get("best_params", {})
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass

    ingestion      = DataIngestionModule(config).run()
    preprocessed   = PreprocessingModule(config).run(ingestion)
    features       = FeatureSelectionModule(config).run(preprocessed)
    classification = ClassificationModule(config).run(features, skip_training=True)

    if best_params:
        classification.cv_results["best_params"] = best_params
        print(f"[run_experiments] Loaded best_params from Experiment 1: {best_params}")
    else:
        print("[run_experiments] WARNING: could not load best_params from results.json; "
              "stability will use default XGBoost hyperparameters. Run main.py first.")

    # Experiment 2 — SHAP coherence and pathway enrichment
    explainability = ExplainabilityModule(config)
    explainability.run(classification)

    # Experiment 3 — SHAP stability across retraining runs
    StabilityModule(config).run(ingestion, classification)

    print("\n[run_experiments] Experiments 2 and 3 finished. Results saved to outputs/.")


if __name__ == "__main__":
    main()
