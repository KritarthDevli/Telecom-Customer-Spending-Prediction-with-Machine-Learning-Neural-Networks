Telecom-Customer-Spending-Prediction-with-Machine-Learning-Neural-Networks

This project analyses telecom customer churn data and predicts **totalCharges** using both classical Machine Learning and Deep Learning models. It demonstrates a complete end-to-end AI workflow: data cleaning, preprocessing, encoding, scaling, model training, and performance comparison.

## 📌 Objectives
- Load and explore a real telecom dataset  
- Clean missing and inconsistent data  
- Encode categorical variables  
- Scale numerical features  
- Train ML and DL regression models  
- Compare model performance using MSE  

## 📁 Dataset Overview
The dataset includes:
- Demographics (gender, seniorCitizen, dependents)  
- Services used (internet, phone, contract type)  
- Billing information (monthlyCharges, totalCharges)  
- Churn-related information  

Target variable: **totalCharges**

## 🧠 Techniques Used
- Pandas & NumPy  
- One-hot encoding  
- StandardScaler  
- Linear Regression (Scikit-Learn)  
- Deep Neural Network (Keras Sequential Model)  
- Mean Squared Error (MSE) evaluation  

## 🚀 Models Implemented
### 1. Linear Regression  
A classical ML regression model used as a baseline.

### 2. Deep Learning Model  
A neural network with:
- Dense(64, ReLU)  
- Dense(32, ReLU)  
- Dense(1) output layer  

Optimized using **Adam** and trained for 50 epochs.

## 📊 Results
Both models predict customer spending, with the deep learning model generally showing better MSE due to capturing nonlinear patterns in the dataset.

