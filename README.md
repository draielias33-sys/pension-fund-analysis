# 🏦 Pension Funds Analysis in Israel (1999-2022) 🇮🇱

This project explores the correlation between popularity and performance in the Israeli pension market using advanced Data Science techniques.

---

## 🎯 Research Goals
* **Wisdom of Crowds?** Do the most popular funds (by assets/deposits) actually deliver the best performance?
* **Clustering Analysis:** Categorizing funds to identify real leaders based on risk-adjusted returns (Sharpe Ratio).

---

## 🧠 Machine Learning Algorithms Used

The project focuses on unsupervised learning to group pension funds using two main approaches:

### 1. Gaussian Mixture Model (GMM) 📊
Used to cluster funds based on statistical distributions. Unlike K-Means, GMM allows for elliptical clusters, which is better suited for financial data.

### 2. Bayesian Gaussian Mixture (BGMM) 🧬
This is the **primary algorithm** used. It utilizes a variational inference process to automatically determine the optimal number of clusters, preventing "over-fitting" of the data.



---

## 🛠️ Installation & Setup

### 1. Prerequisites
Make sure you have **Python 3.8+** installed.

### 2. Install Required Libraries 📦
Run the following command in your terminal:

```bash
pip install pandas numpy scikit-learn
