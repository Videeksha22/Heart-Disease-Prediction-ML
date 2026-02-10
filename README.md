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

## ⚙️ How the Project Works

1. Data Loading
- Clinical heart disease dataset (heart.csv) is loaded
2. Exploratory Data Analysis
- Visualization of feature distributions and correlations
3. Preprocessing
- Feature scaling and train-test split
4. Model Training
- Multiple classifiers are trained and evaluated
5. Model Selection
- KNN selected based on performance metrics
6. Deployment
- Final model integrated into a Streamlit web application

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

```
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
```

---

## 🧪 Model Evaluation

- Multiple machine learning models were evaluated using standard classification metrics
- KNN provided the best balance between accuracy and generalization
- Trained model and preprocessing objects are reused during deployment

---

## 🧑‍💻 Usage

1. Open the Streamlit web app
2. Enter patient health parameters (age, cholesterol, blood pressure, etc.)
3. Click Predict
4. The app outputs whether the person is at risk of heart disease

---

## ▶️ How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/Videeksha22/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML
```

2. Create & activate a virtual environment
   
```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
.\venv\Scripts\activate       # Windows
```

3. Install requirements:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

5. Open the app in browser:

   The terminal will show a local URL (e.g., http://localhost:8501) — open it to interact with the app.

---

## 📌 Key Learnings

* Proper preprocessing and feature scaling significantly improve model performance
* Evaluation metrics like precision and recall are more important than accuracy in healthcare prediction
* Simple models such as KNN can perform well on structured medical data
* Deploying models with Streamlit enables real-time user interaction

---

## 🔗 Live Demo:  

https://heart-disease-prediction-ml-videeksha.streamlit.app/

---

> ⚠️ **Disclaimer:** This project is for educational purposes only and should not be used for medical diagnosis.
