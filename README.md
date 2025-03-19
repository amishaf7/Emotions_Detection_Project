# Emotion Detection using Machine Learning

## Introduction

Emotion detection is a Natural Language Processing (NLP) task that classifies text into different emotional categories. This project builds a machine learning model to predict emotions based on text input. It can be applied in sentiment analysis, customer feedback, and mental health monitoring.

## Features

- Multi-class Emotion Classification (e.g., Happy, Sad, Angry, Surprised, Fearful, Neutral)

- Text Preprocessing (Cleaning, Removing Special Characters, Stopwords)

- Multiple ML Models Evaluated (Naive Bayes, Logistic Regression, Random Forest, SVM)

- Visualizations (Label Distribution, WordCloud, Confusion Matrix)

- Deployment with Streamlit for interactive demo

## Dataset

The dataset contains textual data labeled with emotions. The key columns are:

- text: Input text expressing an emotion.

- label: Numerical representation of the emotion.

The dataset is split into training and testing sets to ensure model generalization.

## Technologies Used

- Python

- Scikit-Learn

- Pandas & NumPy

- Seaborn & Matplotlib

- Streamlit (for deployment)

- Joblib (for model saving)

## Project Structure
    |-- Emotion Detection
    |-- app.py  # Streamlit Application
    |-- emotion_detection.ipynb  # Jupyter Notebook with ML workflow
    |-- best_emotion_model.pkl  # Saved ML model
    |-- vectorizer.pkl  # TF-IDF Vectorizer
    |-- dataset.csv  # Labeled text data

## Installation
Clone the repository and install the required dependencies:
    
    pip install -r requirements.txt
    
## How to Run the Project

1. Train the Model (Run emotion_detection.ipynb to train and save the model)

2. Run the Streamlit App

       streamlit run app.py

3. Access the App in Browser

- Enter text input to detect the emotion!

## Visualization Examples

### Emotion Label Distribution

### WordCloud of Common Words

### Confusion Matrix

### Deployment

The app is deployed using Streamlit and can be accessed online. If running locally, use:

    streamlit run app.py
