# 🧠 Breast Cancer Prediction using Logistic Regression

## 📌 Project Overview
In this project, I built a **Binary Classification Machine Learning model** using **Logistic Regression** to predict whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)**.

I implemented the complete **machine learning workflow**, starting from data preprocessing to model evaluation. The model is trained on the **Breast Cancer Wisconsin Dataset**, which contains multiple numerical measurements of cell nuclei extracted from breast mass images.

The objective of this project was to understand how **Logistic Regression works in real-world classification problems** and how to properly evaluate a machine learning model.

---

# 📊 Dataset Information

**Dataset:** Breast Cancer Wisconsin Dataset

The dataset contains **30 numerical features** computed from digitized images of breast masses.

### Example Features
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension

### Target Variable

| Value | Meaning |
|------|--------|
| 0 | Benign (Non-Cancerous) |
| 1 | Malignant (Cancerous) |

---

# 🛠️ Technologies Used

In this project, I used the following tools and libraries:

- Python 🐍
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

# ⚙️ Machine Learning Workflow

## 1️⃣ Data Loading
I loaded the dataset using **Pandas** and performed initial exploration to understand the structure of the dataset.

```python
import pandas as pd

df = pd.read_csv("data.csv")
df.head()
```

---

## 2️⃣ Data Cleaning

To prepare the dataset for training, I performed the following preprocessing steps:

- Removed unnecessary columns such as `id` and `Unnamed: 32`
- Checked for missing values
- Converted the categorical target variable into numerical form

```
M → 1 (Malignant)
B → 0 (Benign)
```

---

## 3️⃣ Feature Scaling

Since the dataset features have **different scales**, I normalized them using **StandardScaler** to ensure all features contribute equally to the model.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 4️⃣ Train-Test Split

I split the dataset into **training and testing sets** to evaluate the model on unseen data.

- **70% Training Data**
- **30% Testing Data**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
```

---

## 5️⃣ Model Training

I trained a **Logistic Regression model** using Scikit-Learn.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

---

## 6️⃣ Model Prediction

After training the model, I used it to make predictions on the test dataset.

```python
predictions = model.predict(X_test)
```

---

# 📈 Model Evaluation

To evaluate the performance of the model, I used several classification metrics.

### Accuracy

The model achieved approximately:

```
Accuracy: ~98%
```

---

## Classification Report

I used a **classification report** to analyze the model performance in terms of:

- Precision
- Recall
- F1-score
- Support

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
```

These metrics provide deeper insights into how well the model performs on each class.

---

# 📉 Confusion Matrix

I also used a **confusion matrix** to visualize the model’s prediction results.

| Actual / Predicted | Benign | Malignant |
|--------------------|--------|-----------|
| Benign             | True Negative | False Positive |
| Malignant          | False Negative | True Positive |

This helps understand **false positives and false negatives** in the model predictions.

---

# 📚 Key Concepts Covered

Through this project, I practiced and implemented the following concepts:

- Binary Classification
- Logistic Regression
- Sigmoid Function
- Feature Scaling
- Train-Test Split
- Model Evaluation Metrics
- Confusion Matrix

---

# 📁 Project Structure

```
Breast-Cancer-Logistic-Regression/
│
├── Breast_Cancer_Logistic_Regression.ipynb
├── data.csv
└── README.md
```

---

# 💡 Learning Outcomes

By completing this project, I gained hands-on experience in:

- Building a **Logistic Regression classification model**
- Performing **data preprocessing and feature scaling**
- Training and evaluating machine learning models
- Interpreting **precision, recall, and F1-score**
- Understanding the **complete machine learning pipeline**

---

# 🔮 Future Improvements

In the future, I plan to improve this project by:

- Performing **hyperparameter tuning**
- Applying **cross-validation**
- Comparing with other classification models (SVM, Random Forest)
- Deploying the model using **Flask or Streamlit**
