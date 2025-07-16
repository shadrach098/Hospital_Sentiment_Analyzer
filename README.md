# Hospital_Sentiment_Analyzer

# 🌐 Sentiment Analysis in Healthcare: Analyzing Patient Feedback

## 🔍 Introduction

This project focuses on analyzing patient feedback using sentiment analysis. It aims to classify sentiments (positive or negative) in textual data collected from hospital reviews. The integration of **AutoML (TPOT)** helps automate model selection and hyperparameter tuning to enhance accuracy with minimal manual effort.

### 🏛️ Problem Statement

* **Context**: Understanding sentiments in patient feedback.
* **Challenge**: Traditional ML approaches require time-intensive tuning.
* **Solution**: Use TPOT AutoML for automated, optimized model pipelines.
* **Impact**: Faster, scalable, and more accurate feedback classification.

---

## 🤝 Machine Learning Prediction & Outcomes

### 🔬 Research Insights

* Sentiment analysis improves service delivery and patient satisfaction.
* ML models like Logistic Regression and Random Forest perform well.
* AutoML tools like TPOT reduce manual tuning efforts significantly.

### 🎯 Project Goal

Build an ML pipeline that classifies patient sentiment as either **positive** or **negative**.

### ⚖️ Prediction vs. Expectation

* **Expected**: TPOT will outperform manually selected models.
* **Observed**: Traditional models gave moderate results; TPOT offers better accuracy.

### 📊 Dataset Summary

* **Total Records**: 996
* **Sentiment Split**: 73% Positive, 27% Negative
* **Rating Range**: Mostly skewed toward 4 and 5
* **Action Taken**: Applied **SMOTE** to balance classes.

---

## 📊 Data Preprocessing

* Cleaned text (removed punctuation, stopwords)
* Tokenization
* Lemmatization
* TF-IDF Vectorization

---

## 🧮 Machine Learning Models Used

| Model                  | Description                          |
| ---------------------- | ------------------------------------ |
| TPOT Classifier        | AutoML pipeline selection and tuning |
| RandomForestClassifier | Traditional ensemble learning        |
| DecisionTreeClassifier | Tree-based classification            |
| MultinomialNB          | Naive Bayes for discrete features    |

### Resampling

* **SMOTE** used to handle class imbalance.

---

## 📊 Results

| Model                  | Accuracy    |
| ---------------------- | ----------- |
| RandomForestClassifier | 0.83%        |
| DecisionTreeClassifier | 0.79%        |
| MultinomialNB          | 0.80%        |
| TPOT (Best)            | 84%          |

> *(Actual scores plotted in final visualizations)*

### Final TPOT Pipeline

The best model pipeline was selected and exported as `.pk1` and `.pt` files.

---

## 📊 Visualizations

* Sentiment distribution
* Ratings histogram
* Word cloud of patient feedback
* Model accuracy comparison bar chart

---

## 🌐 Live Prediction Example

```python
Comment = {
  1: "Postive Feedback. Thank you so much!",
  0: "Negative Feedback. We're sorry and will improve."
}
```

Example predictions:

* **Positive**: "Wow! Great experience. Staff and doctors were amazing."
* **Negative**: "I hate this place. I wouldn't recommend it."

---

## 🎓 References

* Georgian College Lectures
* Google Cloud AutoML Documentation
* Kaggle Hospital Feedback Dataset
* Liu, B. (2012). Sentiment Analysis and Opinion Mining
* TPOT: [https://epistasislab.github.io/tpot/](https://epistasislab.github.io/tpot/)
* Zhou et al. (2020). Sentiment Analysis in Biomedical Texts

---

## 🤜 How to Use

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook or script
4. Replace sample input with actual hospital feedback

---

## 💼 Author

**Bruce-Arhin Shadrach**
AI & ML Engineer | Georgian College
[LinkedIn](https://www.linkedin.com/in/bruce-arhin-shadrach/) | [brucearhin098@gmail.com](mailto:brucearhin098@gmail.com)
