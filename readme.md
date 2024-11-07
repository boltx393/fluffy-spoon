# NBA Game Prediction and Clustering

This project aims to predict home team wins in NBA games using classification models and analyze patterns in the data using clustering techniques.

## Table of Contents

- [Overview](#overview)
- [Data Preprocessing](#data-preprocessing)
- [Models](#models)
  - [Random Forest Classification](#random-forest-classification)
  - [Logistic Regression](#logistic-regression)
  - [K-Means Clustering](#k-means-clustering)
- [Visualizations](#visualizations)
- [Performance Analysis](#performance-analysis)
- [Installation](#installation)
- [Usage](#usage)

## Overview

This repository contains an analysis of NBA games, where we predict the outcome of home games and identify patterns within the data using machine learning techniques. The main tasks include:

1. Data preprocessing
2. Predicting game outcomes using Random Forest and Logistic Regression
3. Grouping similar games with K-Means clustering
4. Evaluating model performance using various metrics
5. Generating visualizations for better insights

## Data Preprocessing

The dataset includes information on NBA games, players, and teams. Key preprocessing steps include:

- Loaded data from various sources (games, games_details, players, teams)
- Handled missing values using **SimpleImputer**
- Applied feature scaling using **StandardScaler**
- Created new features such as point differentials and rolling averages

## Models

### Random Forest Classification

- Predicted home team wins
- Tuned hyperparameters using **grid search**
- Evaluated model using:
  - **Accuracy**, **Precision**, **Recall**, **F1 Score**
  - **ROC AUC Score**
  - **Cross-validation scores**
  - **Feature importance analysis**
  - **Confusion matrix visualization**

### Logistic Regression

- Used as a second classification model for comparison
- Optimized hyperparameters using **grid search**
- Evaluated model using:
  - **Classification report**
  - **ROC AUC score**
  - **Accuracy metrics**
  - **Cross-validation results**

### K-Means Clustering

- Grouped similar games together
- Determined optimal clusters using:
  - **Silhouette score**
  - **Calinski-Harabasz score**
  - **Elbow method**
- Visualizations included:
  - **Cluster evaluation plots**
  - **Cluster size distribution**
  - **PCA-based cluster visualization**

## Visualizations

- **Random Forest confusion matrix**
- **Clustering evaluation metrics**
- **Cluster distributions**
- **PCA-based cluster visualization**

## Performance Analysis

Comprehensive analysis for all models including:

- Performance metrics (accuracy, precision, recall, F1 score, ROC AUC)
- **Cross-validation results**
- **Feature importance rankings**
- **Cluster characteristics analysis**

## Installation

To run this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/nba-game-prediction.git
cd nba-game-prediction
pip install -r requirements.txt
```

## Usage

To run the project:

1. Ensure you have Python 3.x installed.
2. Run the script for data preprocessing, model training, and evaluation:
   ```bash
   python test.py
   ```
   This will:
   - Preprocess the data
   - Train and evaluate the models
   - Display visualizations
