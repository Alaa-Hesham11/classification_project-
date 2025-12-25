
# Salary Classification using Multiple Machine Learning Models

## 📌 Project Description
This project implements a complete **machine learning classification pipeline** to predict **salary classes** based on structured tabular data.  
The notebook covers **data exploration, preprocessing, training, evaluation, hyperparameter tuning**, and **model comparison** using **six different classification models**.

The final goal is to **identify the best-performing model** based on evaluation metrics.

---

## 📂 Dataset
- Input format: **CSV file**
- Dataset is uploaded interactively using **Google Colab**
- Target variable: **`salary`**

> ⚠️ Dataset file is not included in this repository.

---

## 🔍 Data Exploration
The notebook performs:
- Dataset shape inspection (rows & columns)
- Column names and data types
- Missing value detection
- Unique value analysis for categorical features (before encoding)

---

## 🛠️ Data Preprocessing
Steps applied:
- Column name cleaning
- Handling missing values:
  - Categorical features → filled with mode
  - Numerical features → filled with median
- Encoding categorical variables using **Label Encoding**
- Feature scaling using **StandardScaler**
- Train-test split (80% training, 20% testing)

---

## 🤖 Machine Learning Models Used
The following **6 classification models** were trained and evaluated:

1. Logistic Regression  
2. Random Forest Classifier  
3. K-Nearest Neighbors (KNN)  
4. Decision Tree Classifier  
5. Naive Bayes (GaussianNB)  
6. Linear Support Vector Machine (LinearSVC)

---

## 📊 Model Evaluation
Each model is evaluated using:
- **Accuracy**
- **F1-Score (weighted)**

The results are collected and compared in a summary table.

---

## 🔧 Hyperparameter Tuning
- **GridSearchCV** is applied to all applicable models
- Cross-validation: `cv = 3`
- Scoring metric: `f1_weighted`

### Tuned Models:
- Logistic Regression
- Random Forest
- KNN
- Decision Tree
- Linear SVM

> Naive Bayes is evaluated without tuning due to limited hyperparameters.

For each tuned model:
- Best hyperparameters are selected
- Performance is re-evaluated on the test set

---

## 🏆 Best Model Selection
- Models are ranked based on **F1-Score**
- The **best-performing model** is selected and reported
- Comparison is provided **before and after tuning**

---

## 🧪 Libraries & Tools
- Python
- Pandas
- NumPy
- Scikit-learn
- Google Colab

---

## ▶️ How to Run
1. Open the notebook in **Google Colab**
2. Upload the dataset when prompted
3. Run all cells sequentially

---

## 📌 Notes
- Large datasets and trained models are not stored in the repository
- This notebook is suitable for:
  - Academic projects
  - Machine learning practice
  - Model comparison studies

---

## ✨ Author
**Alaa Hesham**
