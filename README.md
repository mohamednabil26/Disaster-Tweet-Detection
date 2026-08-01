# Disaster-Tweet-Detection
# 🚀 Bilingual NLP Text Classification Pipeline (English & Arabic)

An end-to-end, production-oriented Natural Language Processing (NLP) project focused on classifying **English Disaster Tweets** and **Arabic Dialectal Text**. This project explores the full NLP lifecycle—from exploratory data analysis and custom preprocessing pipelines to model optimization and failure mode analysis.

---

## 📌 Project Highlights

* **Bilingual Text Processing:** Custom cleaning and normalization routines built for both English (HTML unescaping, regex tokenization) and Arabic (diacritics removal, letter normalization, noise reduction).
* **Ablation Studies:** Systematic evaluation of standard NLP techniques—comparing Raw vs. Cleaned text, Stopword Removal, Stemming (Porter & ISRI), Lemmatization, and Sub-word Character $n$-grams (`char_wb`).
* **Model Exploration:** Rigorous comparison across Naive Bayes, Logistic Regression, and Linear Support Vector Classifiers (LinearSVC).
* **Class Imbalance & Threshold Tuning:** Optimizing decision thresholds and using class-weight balancing to maximize **Recall** for emergency response detection and **Macro F1** for Arabic multi-class distribution.
* **Error Analysis:** Categorization of model failure modes including metaphors, sarcasm, out-of-vocabulary entities, and label noise.

---

## 🛠️ Tech Stack & Tools

* **Language:** Python
* **Machine Learning & NLP:** `scikit-learn`, `nltk`
* **Data Processing & Analytics:** `pandas`, `numpy`
* **Techniques:** TF-IDF Vectorization, Character $n$-grams (`char_wb`), GridSearchCV, Pipeline Architecture

---

## 📊 Key Results

* **English Baseline vs. Final:** Improved F1-Score from **~0.75** (Baseline) to **>0.80** using tuned TF-IDF + Logistic Regression with custom text cleaning.
* **Decision Thresholding:** Adjusted decision boundary to `0.35` to significantly minimize False Negatives (Missed Disasters) for real-world deployment safety.
* **Arabic Insights:** Character-level $n$-grams (`char_wb (2,4)`) significantly outperformed standard word-level models and aggressive Arabic stemming (ISRI) by preserving morphological roots without loss of contextual semantics.

---

## 📂 Repository Structure

```text
├── data/                  # Dataset files (English & Arabic)
├── notebook.ipynb         # Complete notebook containing Phase 0 to Phase 7
└── README.md              # Project documentation
