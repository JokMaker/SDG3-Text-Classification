# SDG 3 Indicator Text Classification

**MLT1 Formative 2 - Group Assignment | African Leadership University**

**Group 6** - Jok John Maker Kur · Nanen Miracle Mbanaade · Nziza Aime Pacifique

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Experiments Summary](#experiments-summary)
- [Key Findings](#key-findings)
- [Group Contributions](#group-contributions)
- [Links](#links)

---

## Overview

This project builds a **multi-label text classification system** that predicts which indicators of **Sustainable Development Goal 3 (Good Health and Well-Being)** are relevant to a given text document.

Documents in the dataset include tenders, reports, humanitarian initiatives, news articles, and development organisation publications. Each document can relate to **multiple SDG 3 indicators simultaneously**, making this a multi-label NLP problem.

**The pipeline covers:**

- Exploratory data analysis and class imbalance investigation
- Text preprocessing (HTML stripping, lemmatisation, stop word removal)
- TF-IDF feature engineering (unigrams and bigrams)
- 11 systematic experiments across model types, balancing strategies, thresholds, and embeddings
- Final model selection and test set inference
- Submission file generation

| Metric | Value |
|--------|-------|
| Evaluation metric | Hamming Loss (lower = better) |
| Best validation Hamming Loss | `0.0463` - TF-IDF Bigrams + LinearSVC (C=10.0)

---

## Dataset

| File | Description | Rows |
|------|-------------|:----:|
| `Devex_train.csv` | Training data with labels (Label 1–12) | 2,995 |
| `Devex_test_questions.csv` | Test data without labels | 998 |

**Label structure:** 27 unique SDG 3 indicators (e.g. `3.1.1 - Maternal mortality ratio`, `3.3.1 - HIV infections`, `3.b.2 - ODA to medical research`)

**Key challenge:** Severe class imbalance - most frequent label appears 1,044 times, rarest appears 165 times (6.3× ratio after binarisation)

Download the datasets from the Canvas assignment page and place them in:

```
SDG3-Assignment/
└── data/
    ├── Devex_train.csv
    └── Devex_test_questions.csv
```

---

## Project Structure

```
SDG3-Assignment/
├── data/
│   ├── Devex_train.csv
│   └── Devex_test_questions.csv
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb     ← EDA, cleaning, label binarisation
│   ├── 02_Baseline_Experiments.ipynb      ← 11 experiments, evaluation
│   └── 03_Final_Model_Inference.ipynb     ← final model, submission file
│
├── outputs/
│   ├── train_clean.csv                    ← preprocessed training data
│   ├── test_clean.csv                     ← preprocessed test data
│   ├── all_indicators.json                ← list of 27 SDG 3 indicators
│   ├── experiment_results.csv             ← all 11 experiment metrics
│   ├── per_label_f1.csv                   ← per-label performance breakdown
│   ├── best_thresholds_e9.npy             ← saved per-label thresholds
│   ├── X_train_sbert.npy                  ← SBERT training embeddings (E11)
│   ├── X_val_sbert.npy                    ← SBERT validation embeddings (E11)
│   ├── submission.csv                     ← final test predictions (998 rows)
│   └── figures/
│       ├── document_type_distribution.png
│       ├── label_distribution.png
│       ├── labels_per_sample.png
│       ├── text_length_distribution.png
│       ├── wordcloud_all.png
│       ├── top_terms_per_label.png
│       ├── doctype_vs_label.png
│       ├── label_cooccurrence_heatmap.png
│       ├── preprocessing_effect.png
│       ├── experiment_comparison.png
│       ├── learning_curves_sgd.png
│       ├── per_label_confusion_heatmap.png
│       ├── hyperparameter_tuning_C.png
│       ├── per_label_f1.png
│       ├── final_per_label_f1.png
│       ├── precision_recall_scatter.png
│       ├── true_vs_predicted_freq.png
│       └── final_experiment_comparison.png
│
└── report/
    └── formative2_group6_Assignment2.pdf
```

---

## How to Run

### Requirements

All notebooks run on **Google Colab** - no local setup needed. The following libraries are installed automatically inside each notebook:

```
scikit-learn
scikit-multilearn
imbalanced-learn
beautifulsoup4
nltk
pandas
numpy
matplotlib
seaborn
wordcloud
sentence-transformers
```

### Steps

**Step 1 - Set up Google Drive**

1. Create a folder called `SDG3-Assignment` in your Google Drive
2. Inside it, create: `data/`, `notebooks/`, `outputs/`
3. Upload `Devex_train.csv` and `Devex_test_questions.csv` into `data/`

**Step 2 - Run Notebook 1**

Open `01_EDA_and_Preprocessing.ipynb` in Colab and run all cells top to bottom.
Produces `train_clean.csv`, `test_clean.csv`, and all EDA figures.

**Step 3 - Run Notebook 2**

Open `02_Baseline_Experiments.ipynb` in Colab and run all cells top to bottom.
Runs all 11 experiments and saves `experiment_results.csv`.

**Step 4 - Run Notebook 3**

Open `03_Final_Model_Inference.ipynb` in Colab and run all cells top to bottom.
Generates `submission.csv` in `outputs/`.

> All three notebooks mount Google Drive automatically. When prompted, authenticate with your Google account and allow access.

---

## Experiments Summary

| Exp | Description | Hamming Loss | F1 Macro |
|:---:|-------------|:------------:|:--------:|
| E1  | TF-IDF unigrams + LR (no balancing) — baseline | 0.0565 | 0.2062 |
| E2  | + class_weight='balanced' | 0.0588 | 0.5475 |
| E3  | TF-IDF bigrams + LR balanced | 0.0571 | 0.5360 |
| E4  | TF-IDF bigrams + LinearSVC balanced  | **0.0463** | 0.4825 |
| E5  | TF-IDF unigrams + Random Forest balanced | 0.0556 | 0.2518 |
| E6  | TF-IDF bigrams + LR + per-label threshold tuning | 0.0547 | **0.5839** |
| E7  | TF-IDF bigrams + SGD (log loss) balanced | 0.0504 | 0.5451 |
| E8  | TF-IDF bigrams + LR + MLSMOTE oversampling | 0.0589 | 0.5386 |
| E9  | Best synthesis: bigrams + LR + threshold tuning | 0.0547 | 0.5839 |
| E9b | LinearSVC hyperparameter tuning (C values) | 0.0463 | 0.4825 |
| E10 | Majority voting ensemble: LR + SVM + SGD | 0.0503 | 0.5411 |
| E11 | SBERT embeddings + LinearSVC | 0.0555 | 0.2526 |

**Final model: E4 - TF-IDF Bigrams + LinearSVC (C=10.0)** (lowest Hamming Loss = 0.0463)

---

## Key Findings

- **Class weighting was the single biggest improvement** - adding `class_weight='balanced'` to E1 improved F1 Macro by 165% (0.2062 → 0.5475) with no other changes
- **SVM outperforms LR on Hamming Loss** but underperforms on F1 Macro - a precision/recall tradeoff
- **Threshold tuning beats MLSMOTE** for rare label recall - synthetic TF-IDF interpolation added noise rather than signal
- **TF-IDF outperforms SBERT** - general-purpose embeddings underperform on domain-specific SDG terminology; BioBERT or PubMedBERT recommended as future improvement
- **Rare labels remain challenging** - indicators with fewer than 15 validation samples (3.1.2, 3.9.3, 3.3.4) still achieve near-zero F1 regardless of balancing strategy

---

## Group Contributions

| Member | Contribution |
|--------|-------------|
| Jok John Maker Kur | Notebook 2 - All experiments (E1–E11), GitHub repo, README |
| Nanen Miracle Mbanaade | Notebook 1 - EDA & Preprocessing, Report: Methodology, Results, Discussion |
| Nziza Aime Pacifique | Notebook 3 - Final Model & Inference, Report: Introduction, Literature, Conclusion, Ethics |

---

## Links

| Resource | Link |
|----------|------|
| GitHub Repository | https://github.com/JokMaker/SDG3-Text-Classification |
| Demo Video | https://drive.google.com/drive/folders/1GcTWb48EqSKzORcR2PlcnIVwoZlP-vXc?usp=drive_link |

---

*MLT1 Formative 2 · African Leadership University · June 2026*
