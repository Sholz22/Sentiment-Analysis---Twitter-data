# 📄 Sentiment Analysis on Twitter Dataset

A Natural Language Processing (NLP) project that builds a sentiment classification model using Twitter data. The project leverages a **custom preprocessing pipeline** and a **Naive Bayes classifier** to predict whether tweets express positive or negative sentiments.


## 🚀 Project Overview

This project demonstrates:

* Cleaning and preprocessing text data using NLTK and custom transformers.
* Building a modular machine learning pipeline using `scikit-learn`.
* Training a sentiment classifier using the Naive Bayes algorithm.
* Evaluating model performance on both training and validation datasets.
* Making real-time sentiment predictions on user-input text.


## 🗂️ Table of Contents

* [Project Overview](#project-overview)
* [Dataset Description](#dataset-description)
* [Project Structure](#project-structure)
* [How to Run](#how-to-run)
* [Model Pipeline](#model-pipeline)
* [Evaluation](#evaluation)
* [Use Cases](#use-cases)
* [Limitations and Improvements](#limitations-and-improvements)
* [License](#license)


## 📚 Dataset Description

* **Training Dataset:** `twitter_training.csv`
* **Validation Dataset:** `twitter_validation.csv`

Each dataset contains:

* `Tweet_ID`: Tweet identifier
* `Entity`: The subject of the tweet
* `Sentiment`: Sentiment label (Positive, Negative, Neutral, Irrelevant)
* `Review`: The tweet text content

For this project:

* Only tweets labeled **Positive** or **Negative** were used.
* Irrelevant and Neutral tweets were excluded.


## 🏗️ Project Structure

```bash
📁 Sentiment_Analysis
│
├── Sentiment_Analysis_Template.ipynb  
├── twitter_training.csv                
├── twitter_validation.csv              
├── requirements.txt   
├── LICENSE 
└── README.md                           
```


## ⚙️ How to Run

```bash
# Clone the repository
git clone https://github.com/Sholz22/Sentiment-Analysis-Twitter-data.git
cd sentiment_analysis_twitter

# Install dependencies
pip install -r requirements.txt

# Run the notebook
Open Sentiment_Analysis_Twitter.ipynb in Jupyter or VS Code
```


## 🛠️ Model Pipeline

1. **Text Preprocessing:**

   * Load and clean the text file by removing stopwords
   * Tokenize the text into individual words using `RegexpTokenizer`
2. **Feature Extraction:**

   * TF-IDF vectorization: Convert the text sequences into numerical features using **TF-IDF Vectorization**.
3. **Model Training:**

   * Train a **Multinomial Naive Bayes classifier**
4. **Evaluation:**

   * Classification Report on test and validation datasets to evaluate the performance of the model
5. **Real-time Prediction:**

   * User-input based sentiment prediction to predict the sentiment of a new user input


## 📊 Evaluation

The model is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**

Performance is reported on both the test split and the validation dataset.

**Test Set Performance (20% Hold-out Data)**

Accuracy = **0.89** 
| Metric    | Negative | Positive |
| --------- | -------- | -------- |
| Precision | 0.88     | 0.91     |
| Recall    | 0.92     | 0.86     |
| F1-Score  | 0.90     | 0.88     |

**Interpretation:**
The model performed well on the hold-out test set, achieving 89% accuracy. It was slightly better at detecting Negative sentiments (higher recall), meaning it correctly identified most of the Negative tweets.

**Validation Set Performance (External Data)**

Accuracy = **0.94** 
| Metric    | Negative | Positive |
| --------- | -------- | -------- |
| Precision | 0.93     | 0.95     |
| Recall    | 0.95     | 0.93     |
| F1-Score  | 0.94     | 0.94     |

**Interpretation:**
The model generalized well to completely unseen data, achieving a 94% accuracy on the external validation set. Both Positive and Negative sentiments were predicted with high precision and recall, indicating strong consistency and reliability.

## 💻 Use Cases

* 🔍 Real-time sentiment tracking of tweets.
* 📈 Brand monitoring and public opinion analysis.
* 💬 Chatbot sentiment understanding.
* 📢 Social media marketing feedback loop.


## ⚡ Limitations and Improvements

### Current Limitations:

* Model may misclassify tweets with sarcasm or complex sentiment.
* Assumes all neutral/irrelevant tweets are not useful for sentiment classification.

### Future Improvements:

* Integrate **real-time Twitter API** to fetch live tweets.
* Apply **deep learning models** like LSTMs or transformers for improved accuracy.
* Incorporate **emoji and hashtag sentiment handling**.

## 📝 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this project with attribution.
