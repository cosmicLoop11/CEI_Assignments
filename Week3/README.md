# Week 3 Assignment - Customer Intelligence System

## Overview

This assignment develops an end-to-end Customer Intelligence System using Machine Learning techniques for country segmentation and prediction.

The goal is to identify meaningful groups of countries based on socio-economic indicators and build models that can predict the segment to which a country belongs.

---

## Assignment Objective

Develop a Customer Intelligence System using:

- Classification
- Ensemble Learning
- Clustering

---

## Dataset

Dataset: Unsupervised Learning on Country Data

Files:

- Country-data.csv
- data-dictionary.csv

The dataset contains socio-economic and health indicators for different countries.

### Features Used

- child_mort
- exports
- health
- imports
- income
- inflation
- life_expec
- total_fer
- gdpp

---

## Assignment Workflow

### 1. Data Understanding

- Loaded the dataset
- Checked shape and data types
- Examined statistical summary
- Verified missing values

### 2. Data Preprocessing

- Removed non-numeric country names for modeling
- Applied StandardScaler for feature scaling

### 3. Clustering

#### K-Means Clustering

- Used the Elbow Method to determine the optimal number of clusters
- Created country segments
- Evaluated clusters using Silhouette Score

#### DBSCAN

- Applied density-based clustering
- Compared clustering behavior with K-Means

### 4. Visualization

- Used PCA (Principal Component Analysis)
- Visualized clusters in two dimensions

### 5. Classification

Generated cluster labels from K-Means and used them as target classes.

Models used:

- Logistic Regression
- Decision Tree

### 6. Ensemble Learning

Models used:

- Random Forest
- XGBoost

### 7. Evaluation

Models were evaluated using:

- Accuracy Score
- Classification Report
- Confusion Matrix
- Feature Importance Analysis

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost

---

## Key Insights

- Countries were successfully grouped into meaningful clusters.
- Developed countries generally showed higher income, GDP per capita, and life expectancy.
- Underdeveloped countries showed higher child mortality and lower economic indicators.
- Random Forest and XGBoost provided strong predictive performance.
- Income, GDP per capita, and child mortality were among the most influential features.

---
