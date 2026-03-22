# Data-Driven & ML Colour Analysis of Portuguese Wines

** Multiscale, Machine Learning, and QSAR Methods Applied to Biomolecules Intensive Course - November 2024 | TCCM Master's Programme**  
Aspet, Toulouse, France  
Assignment submitted to: **Prof. Romuald Poteau** (LPCNO, INSA / Université Toulouse III - Paul Sabatier)

---

## Overview

This project applies machine learning to classify Portuguese Vinho Verde wines as red or white based solely on their physicochemical properties, using an Artificial Neural Network (ANN) built with Keras/TensorFlow. Beyond classification accuracy, the project places emphasis on **model interpretability**, employing SHAP (SHapley Additive exPlanations) values to identify which chemical descriptors most strongly drive the model's predictions — a key step toward transparent, "white-box" ML in chemistry.

The full submitted notebook is included: `Bilge_TCCM_ML_Prof_Poteau_HW.ipynb`

---

## Background & Attribution

This homework was assigned as part of the **Artificial Intelligence and Machine Learning in Chemistry** module delivered by Prof. Romuald Poteau at the Multiscale, Machine Learning, and QSAR Methods Applied to Biomolecules Intensive Course. The module followed a flipped classroom model: participants were required to study three preparatory Jupyter notebooks in advance - covering the Iris database with pandas, statistics and regression, and basics of ANNs for supervised learning - all sourced from Prof. Poteau's open teaching repository:

> [https://github.com/rpoteau/pyPhysChem](https://github.com/rpoteau/pyPhysChem)

The datasets (`winequality-red.csv`, `winequality-white.csv`) are publicly available from the UCI Machine Learning Repository (P. Cortez et al., 2009). The assignment specification (`poteau-hw.docx`) was provided by Prof. Poteau. All modelling, analysis, and interpretability work in the notebook is my own.

---

## Dataset

The two wine quality datasets contain physicochemical measurements for Portuguese Vinho Verde wines:

| Feature | Description |
|---------|-------------|
| Fixed acidity | Tartaric acid concentration |
| Volatile acidity | Acetic acid concentration |
| Citric acid | Citric acid concentration |
| Residual sugar | Sugar remaining after fermentation |
| Chlorides | Salt content |
| Free/total SO₂ | Sulfur dioxide levels |
| Density | Wine density |
| pH | Acidity measure |
| Sulphates | Additive contributing to SO₂ |
| Alcohol | Alcohol percentage |
| Quality | Sensory quality score (not used as target here) |
| **Color** | **Target: 0 = red, 1 = white** |

Combined dataset: 6,497 samples (1,599 red + 4,898 white).

---

## Project Structure & My Contributions

The notebook is organized into eight clearly documented sections:

### Part 1 - Library Installation and Imports
Imports pandas, NumPy, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras, and SHAP. The SHAP library is installed explicitly to enable explainable AI functionality.

### Part 2 - Data Loading and Initial Analysis
Red and white wine datasets are loaded, a binary `color` label is added (0 = red, 1 = white), and the two datasets are concatenated. Correlation heatmaps are produced for each wine type, revealing feature relationships such as the strong negative correlation between alcohol and density in white wines.

### Part 3 - Data Splitting and Preprocessing
An 80/20 train-test split is applied. Six physicochemically meaningful features are selected for modelling: `volatile acidity`, `citric acid`, `residual sugar`, `density`, `sulphates`, and `alcohol`. Features are standardized using `StandardScaler` (zero mean, unit variance) to ensure equal contribution to the model.

### Part 4 - ANN Model Definition and Training
A Sequential Keras model is constructed:

```
Input layer  →  Dense(64, ReLU)  →  Dense(32, ReLU)  →  Dense(1, Sigmoid)
```

Compiled with the Adam optimizer and binary cross-entropy loss. Early stopping (patience = 15, monitoring validation loss) is used to prevent overfitting during up to 300 training epochs.

### Part 5 - Model Evaluation and Performance Analysis
Test accuracy, training/validation history curves, Mean Absolute Error (MAE), and R² are computed and visualized to assess model fit and generalization.

### Part 6 - Confusion Matrix
A confusion matrix is plotted to provide a per-class breakdown of classification performance, revealing the model's reliability across both wine types despite class imbalance (white wines outnumber red approximately 3:1).

### Part 7 - K-Fold Cross-Validation
5-fold cross-validation is performed over the training set to assess model stability. MAE, R², mean accuracy, and standard deviations are tracked across all folds and visualized with error bars, confirming consistent performance and absence of overfitting.

### Part 8 - SHAP Explainability Analysis
SHAP `KernelExplainer` is applied to a background subset of training data to compute feature-level contribution values. SHAP summary plots reveal the relative importance of each physicochemical descriptor in driving the binary classification, providing chemically interpretable insight into what distinguishes red from white wines in the model's learned representation.

---

## Repository Contents

| File | Description |
|------|-------------|
| `Bilge_TCCM_ML_Prof_Poteau_HW.ipynb` | Full submitted Jupyter notebook (my work) |
| `winequality-red.csv` | Red wine dataset (UCI ML Repository) |
| `winequality-white.csv` | White wine dataset (UCI ML Repository) |
| `Poteau_ML_Assignment_LTTC2024.pdf` | Original assignment specification (Prof. Poteau) |

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies (recommended: use a virtual environment)
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow shap jupyter

# Launch the notebook
jupyter lab Bilge_TCCM_ML_Prof_Poteau_HW.ipynb
```

> The datasets must be in the same directory as the notebook, or paths adjusted accordingly.

---

## Key Concepts

- Binary classification with Artificial Neural Networks (ANN)
- Feature selection and standardization for ML pipelines
- Model evaluation: accuracy, MAE, R², confusion matrix
- K-fold cross-validation for generalization assessment
- Explainable AI (XAI) with SHAP values
- Physicochemical data analysis with pandas and seaborn

---

## Academic Context

This work was completed as part of the **Artificial Intelligence & Machine Learning in Chemistry** module at the "Multiscale, Machine Learning and QSAR Methods Applied to Biomolecules", November 2024, held in Aspet, Toulouse, France - an intensive course within the **TCCM European Master's programme** (Theoretical Chemistry and Computational Modelling).

Teaching materials for this module are publicly available at Prof. Poteau's [pyPhysChem repository](https://github.com/rpoteau/pyPhysChem).

*Submitted by: Bilge Emek Cetin - KU Leuven*

---

## Dataset Reference

P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis. *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4):547–553, 2009.
