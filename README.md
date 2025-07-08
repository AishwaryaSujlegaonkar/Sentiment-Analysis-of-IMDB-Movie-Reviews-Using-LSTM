# Sentiment-Analysis-of-IMDB-Movie-Reviews-Using-LSTM

Project Overview : This project focuses on building a deep learning model using LSTM (Long Short-Term Memory) networks to perform sentiment analysis on the IMDB movie reviews dataset. The goal is to classify each review as either positive or negative, leveraging the sequential nature of textual data.

Problem Statement : Develop and evaluate an LSTM-based model to predict the sentiment (positive/negative) of movie reviews from the IMDB dataset.

Skills to learn :
Text preprocessing techniques for NLP
Sequence padding and tokenization
Building deep learning models with LSTM
Model evaluation using accuracy, precision, recall, and F1-score
Working with TensorFlow/Keras for NLP tasks

Project Approach :
1. Dataset Loading :
Load the IMDB dataset from TensorFlow
Explore dataset structure and label distribution

2. Exploratory Data Analysis :
Visualize review lengths
Analyze vocabulary size
Decode sample integer-encoded reviews

3. Preprocessing :
Pad sequences for uniform input length
Optionally decode sequences to readable text

4. Model Building :
Use Embedding layer to convert word indices into dense vectors
Add LSTM layers to capture temporal patterns
Use Dense output layer with sigmoid activation

5. Model Training :
Compile the model with binary_crossentropy loss
Train using training data and validate on the test set

6. Model Evaluation :
Calculate accuracy, precision, recall, and F1-score
Visualize training and validation loss/accuracy curves

7. Predictions :
Test the model on custom reviews
Output sentiment labels ("positive"/"negative")

Evaluation Metrics : Accuracy, Precision, Recall, F1 Score, Binary Cross Entropy Loss

Tools and Technologies : Python, TensorFlow / Keras, Jupyter Notebook, Matplotlib / Seaborn (for visualization)
