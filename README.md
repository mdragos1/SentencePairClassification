# Practical Machine Learning Project

## Project Overview

This project focuses on training a classifier to identify correlations between pairs of Romanian sentences. The dataset used includes 58,000 training pairs, 3,000 validation pairs, and 3,000 test pairs. Two machine learning models were implemented and evaluated: **LinearSVC** and **Neural Networks**.

## Feature Engineering

Two feature extraction methods were used:

- **TF-IDF (Term Frequency - Inverse Document Frequency):** This metric evaluates the importance of a word within a document in relation to a set of documents. It was used to vectorize the concatenated pairs of sentences to capture their significance and potential correlations.

- **Bag of Words (BoW):** This technique represents a document as an unordered set of words, keeping track of word frequency but ignoring grammar and word order.

For both methods, the following parameters were applied:
- **ngram_range:** (1, 2) – unigrams and bigrams
- **strip_accents:** 'unicode' – to remove accents
- **max_features:** 10,000 – to reduce dimensionality for neural network training

## Models Implemented

### 1. LinearSVC
LinearSVC is a variant of the Support Vector Machine designed for binary and multiclass classification using a linear kernel. Various hyperparameters were tuned, including `C` (regularization), `max_iter` (maximum iterations), and `loss` (hinge and squared hinge).

**Key Observations:**
- Best performance was obtained using `C=1`, `loss='hinge'`, with the Bag of Words feature extraction method.
- The model showed limitations in learning well, as indicated by lower metrics for recall, precision, and F1 score compared to accuracy.

### 2. Neural Networks
The neural networks used Dense and Dropout layers with ReLu and Softmax activation functions. The Adam optimizer was employed with different learning rates (0.00001 and 0.0001) and early stopping to prevent overfitting.

**Architectures:**
- **Model 1:** Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.2) → Dense(4, Softmax)
- **Model 2:** Dense(128) → Dropout(0.2) → Dense(4, Softmax)

**Key Observations:**
- Best performance was achieved with Model 1 using Bag of Words features, a learning rate of 0.0001, and patience of 4 for early stopping.
- The models faced challenges with overfitting and imbalanced datasets.

## Conclusion

Both models demonstrated similar performance issues, primarily due to the imbalanced dataset, which affected their ability to classify certain classes accurately. Further improvements could be made with a larger and more balanced dataset.
