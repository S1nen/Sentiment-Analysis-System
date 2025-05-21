# Sentiment Analysis Web Application using XGBoost

## Introduction

This project presents a complete sentiment analysis system capable of classifying text reviews into **positive**, **negative**, or **neutral** sentiments. It integrates a powerful machine learning backend with a clean and responsive web interface, allowing real-time user interaction and classification. The primary goal of this project is to demonstrate how natural language processing (NLP) techniques can be effectively used to extract meaningful insights from unstructured text data and to make these insights accessible through an intuitive web-based interface.

Built using Python’s Flask framework for web development and **XGBoost** as the machine learning engine, the solution leverages modern NLP preprocessing techniques such as **lemmatization**, **stopword removal**, and **CountVectorizer**-based vectorization to transform raw reviews into machine-readable input for prediction. The final application is well-suited for scenarios such as product review analysis, customer feedback monitoring, and opinion mining.

## Problem Statement

In the age of digital commerce and social media, businesses receive massive amounts of textual feedback every day. Manually analyzing each review is not feasible at scale, and there is a growing need for automated systems that can classify and understand the emotional tone of such feedback. This project aims to solve that challenge by automating the process of **sentiment detection** using a trained machine learning model and providing a web interface that anyone can use to test and evaluate sentiments from text reviews.

## Dataset

The project uses a preprocessed subset of the **Amazon Product Reviews Dataset**, which contains thousands of user reviews labeled according to their sentiment polarity. The reviews are labeled as one of the following:

- **Positive** – Favorable opinion about a product or service.
- **Negative** – Dissatisfaction or complaints about a product or service.
- **Neutral** – Reviews that express neither strong satisfaction nor dissatisfaction.

During preprocessing, each review is cleaned by converting it to lowercase, removing punctuation, numbers, and special characters. Then, stopwords (common words with little meaning like "is", "the", "at") are removed. The remaining words are lemmatized (converted to their base/root form) to ensure uniform representation of similar terms.

## Technologies Used

### Programming Language:
- Python – For model development, preprocessing, and Flask server.

### Web Framework:
- Flask – Lightweight web framework used to create and serve the application.

### Natural Language Processing:
- NLTK – Used for tokenization, lemmatization, and stopword filtering.
- Scikit-learn – For text vectorization (CountVectorizer) and model utilities.

### Machine Learning:
- XGBoost – A scalable, accurate gradient-boosting framework used for training the sentiment classifier.
- CountVectorizer – Converts textual data into numerical vectors based on word frequency.
- MinMaxScaler – Used to scale the input features to a normalized range.

### Frontend:
- HTML5 & CSS3 – For creating a simple, responsive user interface.

## How the Model Works

The heart of the application is the **XGBoost classifier**, a high-performance, scalable gradient boosting algorithm. The model is trained on labeled review data using preprocessed and vectorized features. Here's a simplified pipeline of how it works:

1. **Text Preprocessing**: Raw text is cleaned, lemmatized, and stripped of stopwords.
2. **Vectorization**: The cleaned text is transformed into numerical format using `CountVectorizer`, limited to 4000 most frequent tokens.
3. **Scaling**: The vectorized input is scaled to a range between 0 and 1 using `MinMaxScaler`.
4. **Prediction**: The processed data is passed to the trained XGBoost model to predict one of the three sentiment labels.
5. **Output**: The prediction is rendered dynamically on the web interface.

This pipeline is modular and designed for scalability and extension to other text classification tasks.

## How to Use the Application

### Prerequisites

Ensure Python 3.7+ is installed on your system. Then install the required packages.


