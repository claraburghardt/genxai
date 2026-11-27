# GenXAI — Interpretable PAM50 Breast Cancer Subtype Classification

TCC 2026 · Unisinos · Clara Burghardt

---

## Overview

GenXAI is a machine learning pipeline for classifying breast cancer molecular subtypes (PAM50) from RNA-seq gene expression data. It combines XGBoost classification with TreeSHAP explainability and pathway enrichment analysis to produce interpretable predictions.

The pipeline covers three experiments:

| Experiment | Script | Description |
|---|---|---|
| 1 — Classification | `main.py` | Data ingestion → preprocessing → feature selection → XGBoost training/evaluation |
| 2 — SHAP coherence | `run_experiments.py` | SHAP gene attribution + MSigDB/KEGG pathway enrichment per subtype |
| 3 — Stability | `run_experiments.py` | SHAP rank stability across retraining runs (fixed and refit scenarios) |

A Streamlit clinical interface (`app.py`) allows uploading a new sample and obtaining a subtype prediction with SHAP attributions and pathway context.

---

## Project structure

```
genxai/
├── modules/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── classification.py
│   ├── explainability.py
│   └── stability.py
├── data/                  # (gitignored) input datasets
├── outputs/               # (gitignored) model and results — see branch `outputs`
├── main.py                # Experiment 1
├── run_experiments.py     # Experiments 2 and 3
├── app.py                 # Streamlit interface
├── config.yaml            # TCGA-BRCA configuration
├── config_cptac.yaml      # CPTAC-BRCA configuration
└── requirements.txt
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

**Experiment 1** — train and evaluate the classifier:

```bash
python main.py
# or with an alternate config:
python main.py --config config_cptac.yaml
# skip retraining if outputs/xgboost_model.json already exists:
python main.py --skip-training
```

**Experiments 2 and 3** — SHAP coherence and stability (requires Experiment 1 to have run first):

```bash
python run_experiments.py
# or:
python run_experiments.py --config config_cptac.yaml
```

**Clinical interface** — Streamlit app for single-sample prediction:

```bash
streamlit run app.py
```

---

## Data

The pipeline expects two files under `data/`:

- **Expression matrix** — genes × samples (or samples × genes), tab-separated. Configure the orientation with `gene_axis` in `config.yaml`.
- **Labels file** — clinical matrix with a PAM50 subtype column. Configure with `labels_column`.

Supported datasets: TCGA-BRCA (`config.yaml`) and CPTAC-BRCA (`config_cptac.yaml`).

---

## Configuration

All parameters are set in `config.yaml` (or `config_cptac.yaml`). Key options:

| Section | Parameter | Description |
|---|---|---|
| `data` | `drop_normal_like` | `true` → 4-subtype setting (excludes Normal-like) |
| `preprocessing` | `apply_smote` | Oversample minority subtypes during training |
| `feature_selection` | `n_top_genes` | Number of genes selected by mutual information |
| `stability` | `mode` | `fixed` / `refit` / `both` |

---

## Outputs

Results are saved to `outputs/` (gitignored on `main`; available on the `outputs` branch):

- `xgboost_model.json` — trained XGBoost model
- `selected_genes.json` — genes used by the model
- `results.json` — Experiment 1 metrics and CV results
- `results_exp2.json` / `results_exp3_*.json` — Experiments 2 and 3 results
