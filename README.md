# 🫀🩺 Heart Disease Prediction using Machine Learning

This project predicts the risk of heart disease using machine learning techniques and provides an interactive web interface built with Streamlit.  
The repository contains the complete workflow including data analysis, model training, saved models, and a Streamlit application for prediction.

---

## 📌 Project Overview

The goal of this project is to analyze clinical health data and build a machine learning model that can predict whether a person is at risk of heart disease.

The project includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Training and evaluation of machine learning models
- Saving trained models using pickle
- A Streamlit-based web application for user interaction

---

## 📦 Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Joblib

---



## Machine Learning Models

Multiple classification algorithms were explored during training, including:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

The final deployed model is **K-Nearest Neighbors (KNN)** based on evaluation metrics.

---

## 📂 Project Structure

```text
Heart-Disease-Prediction-ML/
│
├── app.py                    # Streamlit web application
├── HeartDisease_Final.ipynb  # Data analysis & model training notebook
├── heart.csv                 # Dataset
│
├── model/
│   ├── knn_heart_model.pkl   # Trained KNN model
│   ├── heart_scaler.pkl      # Scaler used during training
│   └── heart_columns.pkl     # Feature columns
│
├── requirements.txt          # Required Python libraries
└── README.md

---
