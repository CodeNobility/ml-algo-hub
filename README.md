
# **🚀 Machine Learning - Complete Guide**  

Welcome to the **Machine Learning Repository!** This repository contains implementations of various **ML algorithms**, **feature scaling techniques**, and **evaluation metrics** to help you build and optimize machine learning models.  

---

## 📌 **Table of Contents**  
- [Introduction](#introduction)  
- [Types of Machine Learning](#types-of-machine-learning)  
- [Implemented Machine Learning Models](#implemented-machine-learning-models)  
- [Feature Scaling](#feature-scaling)  
- [Evaluation Metrics](#evaluation-metrics)  
- [How to Use This Repository](#how-to-use-this-repository)  
- [Author & Contact](#author--contact)  

---

## 🏁 **Introduction**  
Machine Learning (ML) is a branch of artificial intelligence that enables computers to learn patterns from data and make predictions. It is widely used in fields such as finance, healthcare, marketing, and more.  

This repository serves as a **comprehensive reference** for ML models, best practices, and techniques.  

---

## 🔍 **Types of Machine Learning**  

1️⃣ **Supervised Learning** → Uses labeled data (Classification & Regression).  
2️⃣ **Unsupervised Learning** → Finds patterns in unlabeled data (Clustering, Association Rule Mining).  
3️⃣ **Reinforcement Learning** → Learns through reward-based feedback (e.g., Game AI, Robotics).  

---

## 🚀 **Implemented Machine Learning Models**  

### ✅ **Supervised Learning Models**  
Supervised learning uses **labeled** data, meaning each input has a known output (target variable).  

1️⃣ **Linear Regression**  
   - Used for predicting **continuous values** (e.g., house prices, sales prediction).  
   - Assumes a **linear relationship** between features and target.  

2️⃣ **Logistic Regression**  
   - Used for **binary classification** (Yes/No, Fraud/Not Fraud).  
   - Outputs probabilities using the **sigmoid function**.  

3️⃣ **Decision Tree Classifier**  
   - A tree-like structure where decisions are made based on feature splits.  
   - Works well with **non-linear** data but can overfit.  

4️⃣ **Random Forest Classifier**  
   - An ensemble of multiple decision trees to improve accuracy.  
   - Reduces **overfitting** and performs well on large datasets.  

5️⃣ **Support Vector Machine (SVM)**  
   - Finds the best hyperplane to **separate different classes**.  
   - Works well with high-dimensional data.  

6️⃣ **K-Nearest Neighbors (KNN)**  
   - Stores all training data and classifies new data points based on the **majority vote** of its k-nearest neighbors.  
   - Works well for **small datasets**, but is slow for large ones.  

7️⃣ **Naïve Bayes**  
   - A probabilistic classifier based on **Bayes’ Theorem**.  
   - Works well for **text classification** (spam detection, sentiment analysis).  

---

### ✅ **Unsupervised Learning Models**  
Unsupervised learning is used when **no labeled output** is available. It helps to find hidden patterns or groupings in data.  

1️⃣ **K-Means Clustering**  
   - A **clustering algorithm** that groups similar data points together.  
   - Requires specifying the number of clusters **(K)**.  

2️⃣ **Association Rule Mining (Apriori, FP-Growth)**  
   - Identifies relationships between items in **transactional datasets**.  
   - Used in **market basket analysis** (e.g., “People who buy bread often buy butter”).  

---

## 📊 **Feature Scaling**  

**Why is feature scaling important?**  
Scaling ensures that all features contribute equally to model training.  

**When to use Scaling?**  
| **Algorithm**              | **Scaling Needed?** |
|----------------------------|--------------------|
| Linear Regression          | ✅ Yes  |
| Logistic Regression        | ✅ Yes  |
| Decision Tree              | ❌ No  |
| Random Forest              | ❌ No  |
| SVM                        | ✅ Yes  |
| KNN                        | ✅ Yes  |
| Naïve Bayes                | ❌ No  |
| K-Means                    | ✅ Yes  |
| Association Rule Mining    | ❌ No  |

---

## 📈 **Evaluation Metrics**  

📌 **For Classification Models**  
- **Accuracy**: Overall correctness of the model.  
- **Precision**: How many of the predicted positives are actually positive?  
- **Recall (Sensitivity)**: How well the model captures actual positives?  
- **F1-Score**: Balance between Precision & Recall.  
- **ROC-AUC Score**: Model's ability to distinguish between classes.  

📌 **For Regression Models**  
- **Mean Squared Error (MSE)**  
- **Root Mean Squared Error (RMSE)**  
- **R² Score**  

---

## 🔧 **How to Use This Repository?**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/CodeNobility/machine-learning-guide.git
cd machine-learning-guide
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run Jupyter Notebook**  
```bash
jupyter notebook
```

Explore and experiment with different models inside the notebook! 🚀  

---

## ✍️ **Author & Contact**  

📌 **Name**: Prince Kumar Gupta  
📌 **GitHub**: CodeNobility  
📌 **Email**: princegupta1726@gmail.com

🔥 **Happy Learning & Coding!** 😊🚀  
