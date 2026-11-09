# Original Lab 2 Experiment Results

These are **selected results transcribed from the original console-generated experiment files**, not metrics obtained from an independently rerun portfolio script.

- [Classification model comparison](classification/final_model_comparison.csv): five algorithms tested on 27 held-out utensil images. Accuracy and macro scores are fractions from 0 to 1.
- [Regression model comparison](regression/final_model_comparison.csv): five algorithms tested on 300 held-out car-price records. RMSE and MAE use the original price units.
- Classification: Random Forest had the highest held-out test accuracy (88.89%, 24/27). SVC had the highest validation accuracy (88.89%).
- Regression: Random Forest achieved the lowest test RMSE (84,768.80), and SVR achieved the lowest test MAE (59,094.12).

The original full archive contains hyperparameter tuning CSVs, confusion matrices, model summaries and warnings. Not all original outputs are reproduced here. The original selected SVR had a convergence warning, and the selected Random Forest regressor configuration had effectively identical trees due to bootstrap=False and max_features=None.

**Source:** Fatema Tuj Johora, ENGI 9808 Applied Machine Learning and Neural Networks, Lab 2 (2026). No student number or course-provided dataset is published here.
