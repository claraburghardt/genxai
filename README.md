# GenXAI — Interpretable PAM50 Breast Cancer Subtype Classification

TCC 2026 · Unisinos · Clara Burghardt

---

## Overview

GenXAI is a modular pipeline for PAM50 breast cancer molecular subtype classification from RNA-seq gene expression data. It combines XGBoost classification with TreeSHAP explainability and biological pathway enrichment to produce interpretable predictions, addressing the gap between high-performing genomic models and clinical usability.

The system has five components: data ingestion and preprocessing, feature selection, XGBoost classification, TreeSHAP explainability with pathway mapping, and a Streamlit clinical decision support interface.

The pipeline was developed and evaluated on TCGA-BRCA (844 primary tumor samples, UCSC Xena HiSeqV2) and replicated on an independent cohort, CPTAC-BRCA (122 samples, LinkedOmics).

---

## Experiments

| # | Script | Description |
|---|---|---|
| 1 — Classification | `main.py` | Data ingestion → preprocessing → feature selection → XGBoost training/evaluation |
| 2 — SHAP coherence | `run_experiments.py` | TreeSHAP gene attributions + MSigDB Hallmark 2020 / KEGG 2021 pathway enrichment per subtype |
| 3 — Stability | `run_experiments.py` | SHAP rank stability across 10 retraining runs (fixed panel and re-selection scenarios) |

### Key results

| Metric | TCGA-BRCA | CPTAC-BRCA |
|---|---|---|
| Accuracy | 0.87 | 0.88 |
| AUC-OvR (macro) | 0.981 | 0.996 |
| Macro F1 | 0.78 | 0.84 |
| Biological coherence (Exp. 2) | 4/5 coherent, 1 inconclusive | 2/5 coherent, 3 inconclusive |
| Stability — mean Spearman ρ (Exp. 3) | 0.38 (fixed) / 0.37 (refit) | 0.39 (fixed) / 0.37 (refit) |

The main finding of this work is that good accuracy and plausible explanations are not sufficient: the SHAP gene rankings were unstable across retraining runs (mean ρ ≈ 0.38, well below the 0.85 target), on both cohorts and in both scenarios. This is a limitation that most published work does not test.

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
├── app.py                 # Streamlit clinical interface
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
# with an alternate config:
python main.py --config config_cptac.yaml
# skip retraining if outputs/xgboost_model.json already exists:
python main.py --skip-training
```

**Experiments 2 and 3** — SHAP coherence and stability (requires Experiment 1 to have run first):

```bash
python run_experiments.py
python run_experiments.py --config config_cptac.yaml
```

**Clinical interface** — Streamlit app for single-sample prediction:

```bash
streamlit run app.py
```

---

## Data

| Dataset | Samples | Features | Source |
|---|---|---|---|
| TCGA-BRCA | 844 primary tumors | HiSeqV2 log2-normalized, 20,530 genes | UCSC Xena |
| CPTAC-BRCA | 122 samples | Centered log2 FPKM, PAM50 labels | LinkedOmics |

The pipeline expects two files under `data/`:
- **Expression matrix** — genes × samples (or samples × genes), tab-separated
- **Labels file** — clinical matrix with a PAM50 subtype column

Both are configurable via `config.yaml` / `config_cptac.yaml`.

---

## Configuration

Key parameters in `config.yaml`:

| Section | Parameter | Description |
|---|---|---|
| `data` | `drop_normal_like` | `true` → 4-subtype setting (excludes Normal-like) |
| `preprocessing` | `apply_smote` | Oversample minority subtypes on training set only |
| `feature_selection` | `n_top_genes` | Genes kept by mutual information (default: 500) |
| `stability` | `mode` | `fixed` / `refit` / `both` |

---

## Outputs

Results are saved to `outputs/` (gitignored on `main`; see branch `outputs`):

- `xgboost_model.json` — trained XGBoost model
- `selected_genes.json` — 500 genes used by the model (required by `app.py`)
- `results.json` — Experiment 1 metrics and CV results
- `results_exp2.json` / `results_exp3_*.json` — Experiments 2 and 3 results
