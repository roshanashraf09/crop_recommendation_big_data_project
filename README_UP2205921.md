# Crop Recommendation System — ML Pipeline

## Introduction

This project implements a complete Machine Learning pipeline for a multi-class classification task using the Crop Recommendation Dataset from Kaggle. The system predicts the most suitable crop to grow based on seven soil and climate features: Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall. The dataset contains 2,200 samples across 22 crop types, with each class perfectly balanced at 100 samples. Three classification models were trained, evaluated, and compared — Decision Tree, Random Forest, and Logistic Regression — with hyperparameter optimisation performed via GridSearchCV. The best-performing model (Random Forest) achieved 99.77% accuracy on previously unseen test data.

## Business Objectives

The purpose of this project is to build a reliable crop recommendation system that helps farmers and agricultural planners make data-driven planting decisions based on measurable soil and environmental conditions. The project goal is to achieve a classification accuracy of at least 95% on unseen test data, enabling confident crop recommendations from sensor readings or soil analysis reports.

## ML Pipeline

The ML Pipeline tells the story of this project from data acquisition through to final prediction on unseen data.

### 1. Data Collection

The dataset was obtained from Kaggle: [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset). It can be downloaded programmatically using the Kaggle API:


Data validation checks performed include verifying correct data types (all features are numeric, label is categorical), confirming zero missing values, checking for and removing any duplicate rows, and validating that value ranges are realistic (e.g. pH between 0 and 14, no negative nutrient values).

The dataset was split into three subsets using stratified sampling to preserve class balance: Training set (60%, 1,320 samples), Validation set (20%, 440 samples), and Test set (20%, 440 samples). The test set was held out entirely and only used for final evaluation.

### 2. EDA

Exploratory Data Analysis revealed that the dataset is perfectly balanced with exactly 100 samples per crop class (22 classes, 2,200 total samples). Feature distributions were examined using histograms and found to have varied ranges — for instance, Potassium (K) values range from approximately 5 to 205, while pH values range from 3.5 to 9.9. A correlation matrix showed mostly low inter-feature correlation, with the notable exception of Phosphorus and Potassium (0.74), suggesting these soil nutrients tend to co-occur. Box plots grouped by crop label showed clear separation in feature values for many crops, supporting the hypothesis that classification models should perform well on this data. A pairplot of selected crops confirmed visible clustering across feature combinations.

### 3. Model Building

Three classification models were selected, trained, and compared:

**Model 1 — Decision Tree Classifier:** A single decision tree was chosen as the simplest baseline model. It works by learning a series of yes/no questions about the features to split the data into crop categories. Starting hyperparameters: default (no depth limit, gini criterion). Baseline validation accuracy: 98.18%.

**Model 2 — Random Forest Classifier:** An ensemble of 100 decision trees was chosen to improve on the single Decision Tree. Each tree is trained on a random subset of the data and features, and predictions are made by majority vote. This reduces overfitting and typically improves accuracy. Starting hyperparameters: 100 estimators, default max depth. Baseline validation accuracy: 99.09%.

**Model 3 — Logistic Regression:** A statistical classifier was included as a fundamentally different approach — instead of tree-based splits, it learns mathematical decision boundaries between classes. Features were scaled using StandardScaler since Logistic Regression is sensitive to feature magnitudes. Starting hyperparameters: max_iter=1000, default C=1.0. Baseline validation accuracy: 97.27%.

Model selection was justified by choosing diverse algorithmic approaches (single tree, ensemble of trees, statistical model) to ensure a fair and meaningful comparison.

### 4. Model Evaluation

All three models were evaluated using accuracy score, classification report (precision, recall, F1-score per class), 5-fold cross-validation, and confusion matrices.

Hyperparameter optimisation was performed using GridSearchCV with 3-fold cross-validation for each model:

- **Decision Tree:** Tuned max_depth (None, 5, 10, 15, 20), min_samples_split (2, 5, 10), min_samples_leaf (1, 2, 4), criterion (gini, entropy). Best parameters: entropy criterion, no depth limit. Tuned accuracy: 97.50%.
- **Random Forest:** Tuned n_estimators (100, 200), max_depth (None, 15, 25), min_samples_split (2, 5), min_samples_leaf (1, 2). Best parameters: 200 estimators, no depth limit. Tuned accuracy: 99.32%.
- **Logistic Regression:** Tuned C (0.01, 0.1, 1, 10, 100), solver (lbfgs, newton-cg), penalty (l2). Best parameters: C=100, lbfgs solver. Tuned accuracy: 97.73%.

Baseline and tuned performance were compared side by side. The Random Forest was selected as the best-performing model based on the highest tuned validation accuracy (99.32%). Feature importance analysis revealed humidity and rainfall as the most influential features, followed by Potassium (K) and Phosphorus (P), with pH being the least important.

### 5. Prediction

The best-performing model (Random Forest) was evaluated on the held-out test set (20% of data, completely unseen during training and tuning), achieving 99.77% accuracy (only 1 misclassification out of 440 predictions).

Individual prediction examples were demonstrated using custom input values representing realistic soil and climate conditions:

- Input 1: N=90, P=42, K=43, temp=21°C, humidity=82%, pH=6.5, rainfall=203mm → Predicted: Rice (95.5% confidence)
- Input 2: N=20, P=125, K=200, temp=27°C, humidity=50%, pH=6.8, rainfall=100mm → Predicted: Grapes (38.0% confidence)
- Input 3: N=40, P=60, K=50, temp=30°C, humidity=70%, pH=7.0, rainfall=150mm → Predicted: Papaya (30.5% confidence)

A reusable `predict_crop()` function was created for easy prediction with new inputs.

## Jupyter Notebook Structure

The Jupyter Notebook is organised into the following sections:

1. **Data Collection** — Loading the CSV dataset with pandas
2. **Data Validation** — Type checking, missing values, duplicates, value range validation
3. **Data Preparation** — Label encoding, train/val/test split (60/20/20), feature scaling with StandardScaler
4. **Exploratory Data Analysis** — Class distribution, feature distributions, correlation matrix, box plots by crop, pairplots
5. **Model Building** — Training three baseline models (Decision Tree, Random Forest, Logistic Regression) with cross-validation
6. **Model Evaluation & Hyperparameter Optimisation** — GridSearchCV for all three models, performance comparison chart, confusion matrices, feature importance
7. **Prediction on Unseen Data** — Test set evaluation, individual predictions with confidence scores, reusable prediction function
8. **Summary & Conclusions** — Final results

## Future Work

Given more time, the following improvements could be explored:

- Implementing additional models such as Support Vector Machines (SVM) or Gradient Boosting for comparison
- Using RandomizedSearchCV or Bayesian optimisation (e.g. Optuna) for more efficient hyperparameter search over larger parameter spaces
- Deploying the model as a web application using Flask or Streamlit for real-time crop recommendations
- Incorporating additional features such as soil moisture, sunlight hours, or altitude
- Testing the model on real-world agricultural data from different geographical regions
- Implementing ensemble voting classifiers combining predictions from multiple models

## Libraries and Modules

**pandas** — A data manipulation and analysis library providing DataFrame structures for handling tabular data. Used extensively for loading the CSV dataset, data exploration, statistical summaries, and organising results into comparison tables.

**numpy** — A numerical computing library providing support for arrays and mathematical operations. Used for array manipulations, statistical calculations, and supporting data transformations throughout the pipeline.

**matplotlib** — A comprehensive plotting library for creating static visualisations. Used to generate histograms, bar charts, box plots, and confusion matrix visualisations with customised formatting, labels, and titles.

**seaborn** — A statistical data visualisation library built on matplotlib. Used for the correlation heatmap, pairplots, and applying consistent visual themes across all plots for a professional appearance.

**scikit-learn (sklearn)** — A machine learning library providing tools for classification, preprocessing, model selection, and evaluation. Used for all three classification models (DecisionTreeClassifier, RandomForestClassifier, LogisticRegression), data preprocessing (LabelEncoder, StandardScaler), train/test splitting, GridSearchCV hyperparameter tuning, cross-validation, and evaluation metrics (accuracy, classification report, confusion matrix).

## Unfixed Bugs

No known bugs remain in the project. All code cells execute without errors and produce the expected outputs. Minor sklearn warnings appear during individual predictions due to feature name validation but do not affect results.

## Acknowledgements and References

- Dataset: Atharva Ingle, "Crop Recommendation Dataset", Kaggle, https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
- scikit-learn documentation: https://scikit-learn.org/stable/documentation.html


## Conclusions

The ML pipeline successfully demonstrated that soil and climate features can reliably predict suitable crops. All three models achieved high classification accuracy, with Random Forest performing best at 99.77% on unseen test data — exceeding the 95% accuracy target. Hyperparameter tuning improved Random Forest and Logistic Regression performance, while the Decision Tree showed marginal decrease, suggesting the default parameters were already near-optimal for this dataset. The perfectly balanced dataset contributed to consistent per-class performance, and feature importance analysis revealed humidity and rainfall as the most predictive features — consistent with agricultural domain knowledge. The final model and reusable prediction function provide a practical tool for crop recommendation based on measurable environmental conditions.
