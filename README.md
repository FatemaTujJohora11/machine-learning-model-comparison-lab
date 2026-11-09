# Machine Learning Model Comparison — Lab 2

**Classification and Regression | Applied Machine Learning and Neural Networks**

**Institution:** Memorial University of Newfoundland, Canada  
**Course:** ENGI 9808  
**Status:** Completed academic laboratory project (2026)

## Project Overview

In this laboratory project, I compared five supervised machine learning algorithms for utensil image classification and five for used-car price prediction. I explored the supplied hyperparameter configurations, selected each algorithm's best-performing configuration using validation data, and then evaluated the selected models on separate test sets.

The work is divided into two parts: a three-class image classification problem and a numerical regression problem.

## Part A — Utensil Image Classification

The classification task involved **180 images**, equally distributed across **FORK, KNIFE, and SPOON**. I resized images to **64 × 64 pixels** and represented them using **1,764 Histogram of Oriented Gradients (HOG) features**.

- **Data split:** 126 training / 27 validation / 27 testing images
- **Algorithms:** Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and Support Vector Classifier (SVC)
- **Metrics:** Accuracy, macro precision, macro recall, and macro F1

### Classification Results

| Algorithm | Validation Accuracy | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 81.48% | 62.96% | 0.627 |
| KNN | 81.48% | 70.37% | 0.712 |
| Decision Tree | 74.07% | 48.15% | 0.434 |
| Random Forest | 85.19% | **88.89%** | **0.889** |
| SVC | **88.89%** | 85.19% | 0.850 |

**Key finding:** SVC had the highest validation accuracy, while Random Forest achieved the highest held-out test accuracy (**24 correct predictions out of 27**). With such a small test set, the difference between these models represents just one image, so the ranking should be interpreted cautiously.

## Part B — Used-Car Price Regression

The regression task used **2,000 records** to predict **selling price** from vehicle-related features. It included seven numerical and three categorical input features; categorical variables were encoded and numerical features standardized.

- **Data split:** 1,400 training / 300 validation / 300 testing records
- **Algorithms:** Linear Regression, KNN Regressor, Decision Tree Regressor, Random Forest Regressor, and Support Vector Regression (SVR)
- **Metrics:** Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R²

### Regression Results

| Algorithm | Validation RMSE | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|
| Linear Regression | 121,981.76 | 120,811.99 | 91,399.26 | 0.7500 |
| KNN | 94,000.10 | 87,922.95 | 63,130.31 | 0.8676 |
| Decision Tree | 84,018.53 | 88,392.52 | 64,241.21 | 0.8662 |
| Random Forest | **80,430.76** | **84,768.80** | 61,891.25 | **0.8769** |
| SVR | 87,155.55 | 85,701.64 | **59,094.12** | 0.8742 |

**Key finding:** Random Forest produced the lowest test RMSE, while SVR achieved the lowest test MAE. These measures emphasize different aspects of prediction error.

## Original Experiment Output Files

The selected comparison CSVs come from my original console experiments, not from a new portfolio rerun:

- [Classification: Final Test Results](results/classification/final_model_comparison.csv)
- [Regression: Final Test Results](results/regression/final_model_comparison.csv)
- [Results Notes and Limitations](results/README.md)

The repository does not yet include a complete reproducible implementation of every original hyperparameter sweep.

## Methods and Skills Demonstrated

- Dataset preparation and train/validation/test splitting
- HOG feature extraction for grayscale image classification
- Feature standardization and categorical encoding
- Hyperparameter exploration and validation-based model selection
- Evaluation using confusion matrices, accuracy, precision, recall, F1, RMSE, MAE, and R²
- Comparing generalization performance across multiple algorithms
- Documenting unsuccessful configurations and convergence warnings

## Limitations and Reproducibility

The classification test set contains only 27 images, so the performance rankings have substantial sampling uncertainty. Some classification configurations were incompatible with the original **Python 3.7.16 / scikit-learn 1.0.2** environment. The selected SVR reached its solver iteration limit and issued a convergence warning.

The selected Random Forest regressor used `bootstrap=False` and `max_features=None`, producing essentially identical trees rather than a diverse ensemble. I therefore do not attribute its performance advantage to ensemble diversity.

The original study's results are reported above. Source code and authorized, privacy-reviewed reproducibility materials will be added separately; the metrics should not be interpreted as having been independently reproduced from this GitHub repository yet.

## Author

**Fatema Tuj Johora**  
Master of Applied Science (MASc), Computer Engineering  
Memorial University of Newfoundland, Canada

[GitHub Profile](https://github.com/FatemaTujJohora11) | [LinkedIn](https://www.linkedin.com/in/fatema-tuj-johora/)
