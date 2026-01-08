# AutoJudge: Predicting Programming Problem Difficulty

## 📌 Project Overview
AutoJudge is a machine learning based system that automatically predicts the difficulty of programming problems using only their textual descriptions.  
The system predicts:
- **Difficulty Class**: Easy / Medium / Hard (Classification)
- **Difficulty Score**: Numerical value (Regression)

This project removes the need for manual difficulty labeling and relies purely on Natural Language Processing (NLP) techniques.

---

## 📊 Dataset
The dataset is taken from the **TaskComplexityEval-24** repository and contains programming problems with:
- Title
- Problem Description
- Input Description
- Output Description
- Sample Input/Output
- Difficulty Class
- Difficulty Score

Format used: **JSON Lines (.jsonl)**

---

## 🧹 Data Preprocessing
- Missing text values are handled.
- All text fields are combined into a single feature.
- Sample Input/Output lists are converted into text.
- Only textual information is used (no metadata or user statistics).

---

## 🔍 Feature Extraction
- **TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert text into numerical features.
- Stopwords are removed.
- Maximum features used: 5000.

---

## 🤖 Machine Learning Models

### 🔹 Classification Model
- Algorithm: **Logistic Regression**
- Task: Predict Easy / Medium / Hard
- Evaluation Metric: Accuracy, Confusion Matrix

### 🔹 Regression Model
- Algorithm: **Random Forest Regressor**
- Task: Predict numerical difficulty score
- Evaluation Metrics: MAE, RMSE

---

## 🌐 Web Application
- Built using **Streamlit**
- Users can paste problem details and click **Predict**
- The app displays:
  - Predicted Difficulty Class
  - Predicted Difficulty Score

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
