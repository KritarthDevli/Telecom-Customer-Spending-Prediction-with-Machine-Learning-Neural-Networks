📘 Credit Scoring Income Prediction using Decision Trees & Neural Networks
This project predicts customer total income using a real-world credit scoring dataset. It demonstrates a complete machine learning workflow including data cleaning, preprocessing, feature engineering, classical ML, deep learning, and model comparison.
The models used are Decision Tree Regression and a Keras Neural Network, with performance evaluated using the R² score.

📌 Project Objectives

Load and explore raw credit scoring data
Clean duplicates and missing entries
Encode categorical features
Scale numerical variables
Train a Decision Tree Regressor
Train a Deep Neural Network (Keras Sequential)
Visualize Actual vs Predicted income
Compare the performance of both models

📁 Dataset Overview

The dataset credit_scoring_pre.csv includes:
Feature	Description
age	Customer age
gender	Male/Female
education	Education level
family_status	Marital status
children	Number of children
income_type	Employment category
total_income	Target variable
purpose	Reason for loan
purpose_short	Processed loan purpose

The dataset has mixed numerical and categorical features, requiring preprocessing before modelling.

🔧 Tech Stack

Python
Pandas / NumPy
Scikit-Learn
TensorFlow / Keras
Matplotlib
Jupyter Notebook

🧠 Machine Learning Models
1️⃣ Decision Tree Regressor

A non-linear, tree-based model that handles complex relationships and does not require heavy scaling.

2️⃣ Neural Network (Keras)

A multi-layer perceptron model:

Dense(64, ReLU)
Dense(32, ReLU)
Dense(1) output layer

Optimized with Adam and trained using MSE as loss function.

📊 Model Evaluation

The models are evaluated using:

R² Score
Measures how well the model explains the variance in data
Closer to 1 → better performance
Negative → poor model fit
Visualizations
Scatter plots of:
Actual vs Predicted (Decision Tree)
Actual vs Predicted (Neural Network)
