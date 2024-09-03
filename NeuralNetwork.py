import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, precision_score
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score, precision_score, recall_score
from matplotlib import pyplot as pl
import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def get_features(vectorizer='tfidf'):
    """
    This function extracts features from sentences using TF-IDF or BoW. 
    """
    if(vectorizer == 'tf_idf'):
        vectorizer = TfidfVectorizer(
            max_features = 10000,
            strip_accents = 'unicode',
            analyzer = 'word',
            ngram_range = (1, 2))
    else:
        vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            analyzer='word',
            strip_accents = 'unicode',
            max_features=10000)

    train_data = vectorizer.fit_transform(sentences[:58114])
    valid_data = vectorizer.transform(sentences[58114:61173])
    test_data = vectorizer.transform(sentences[61173:])

    train_data = np.asarray(train_data.toarray())
    valid_data = np.asarray(valid_data.toarray())
    test_data = np.asarray(test_data.toarray())

    return train_data, valid_data, test_data

def plot_confusion_matrix(y_true, predicted, title, labels=None, normalize=None):
    """
    This function plots a confusion matrix with 6x6 size and a 100 dpi resolution.
    """

    # Changed the figure size and increased dpi for better resolution
    _, axes = plt.subplots(figsize=(6,6), dpi=100)

    conf_matrix = confusion_matrix(y_true, predicted, normalize=normalize)
    display = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)

    #I set the title using the axes object.
    axes.set(title=f'Confusion Matrix {title} Model')

    display.plot(ax=axes)
    return conf_matrix

def get_one_hot_encode(train_y, valid_y):
    """
    This function one hot encodes the training and validation labels (e.g.
    [1,3] -> [[0, 1, 0, 0], [0, 0, 0, 1]]) in order to be used in the
    training of the Neural Network.
    """
    
    #The label encoding step could be optional as the labels are already integers from 0 to 3.
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    train_y_encoded = label_encoder.fit_transform(train_y)    
    train_y_encoded = train_y_encoded.reshape(len(integer_encoded), 1)
    train_onehot_encoded = onehot_encoder.fit_transform(train_y_encoded)

    valid_y_encoded = label_encoder.fit_transform(valid_y)
    integer_encoded = valid_y_encoded.reshape(len(valid_y_encoded), 1)
    valid_onehot_encoded = onehot_encoder.fit_transform(valid_y_encoded)

    return train_onehot_encoded, valid_onehot_encoded

def macro_f1_score(y_true, y_pred):
    """
    This function creates a wrapper for the F1 Macro Score in order to
    be used as e metric when training the Neural Network.
    """
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)

    def f1_score_wrapper(y_true_np, y_pred_np):
        return f1_score(y_true_np, y_pred_np, average='macro')

    return tf.py_function(f1_score_wrapper, (y_true, y_pred), tf.float32)

def plot_loss(history):
    """
    This function makes use of the history of the Neural Network training in
    order to plot on the same graph the Training Loss and the Validation Loss.
    """
    pl.title('Loss')
    pl.plot(history.history['loss'], label='train')
    pl.plot(history.history['val_loss'], label='validation')
    pl.legend() 

def get_model(train_data, valid_data, train_onehot_encoded, valid_onehot_encoded, model=1, lr=0.0001, patience=2):
    """
    This function initialises one of the two Neural Networks and trains it for
    a maximum of 20 epochs with a Adam optimizer, Macro F1 metric and the other
    specified paramaters. Finally, returns the trained model.
    """
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    ea = EarlyStopping(patience=patience)
    if model == 1:
        model = Sequential()
        model.add(Dense(256, activation = "relu"))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation = "relu"))
        model.add(Dropout(0.2))
        model.add(Dense(4, activation = "softmax"))
        model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = [macro_f1_score])

    elif model == 2:
        model = Sequential()
        model.add(Dense(128, activation = "relu"))
        model.add(Dropout(0.2))
        model.add(Dense(4, activation = "softmax"))
        model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = [macro_f1_score])

    history = model.fit(np.array(train_data), train_onehot_encoded, validation_data = (np.array(valid_data), valid_onehot_encoded)
            , epochs = 20, callbacks=[ea])
    plot_loss(history)
    return model

def compute_model(train_data, valid_data, train_onehot_encoded, valid_onehot_encoded, model=1, lr=0.0001, patience=2):
    """
    This function gets a model and uses it to make predictions on the 
    validation data. Finally returns the model and the predictions.
    """
    model = get_model(train_data, valid_data, train_onehot_encoded, valid_onehot_encoded, model=model, lr=lr, patience=patience)
    predicted = model.predict(np.array(valid_data))
    valid_preds = [np.argmax(line) for line in predicted]
    return model, valid_preds

def compute_results(valid_y, preds):
    """
    This function computes the performance metrics and prints them on the screen.
    """
    accuracy = round(100*accuracy_score(valid_y, preds), 2)
    f1 = round(100*f1_score(valid_y, preds, average='macro'), 2)
    recall = round(100*recall_score(valid_y, preds, average='macro'), 2)
    precision = round(100*precision_score(valid_y, preds, average='macro'), 2)
    print(f"F1: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    clases = ['class 0', 'class 1', 'class 2', 'class 3']
    print(classification_report(valid_y, preds, target_names=clases))

def write_test_predictions(test_data, model, filename="Submission.csv"):
    """
    This function creates a .csv file with the model predictions on test data.
    """
    test_preds = model.predict(test_data)
    test_guids = df_test['guid']
    columns = ['guid', 'label']
    rows = [ [test_guids.iloc[index], int(test_preds[index])] for index in range(len(test_guids))]

    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerows(rows)

df_train = pd.read_json(f'{os.getcwd()}/train.json', lines=False)
df_valid = pd.read_json(f'{os.getcwd()}/validation.json', lines=False)
df_test = pd.read_json(f'{os.getcwd()}/test.json', lines=False)
df = pd.concat([df_train, df_valid, df_test])

s1 = df['sentence1'].tolist()
s2 = df['sentence2'].tolist()

#In this part I concatenate the sentences with a '\n' between them before putting them into the vectorizer (i.e. TF-IDF or BoW)
sentences = [(f'{df["sentence1"].iloc[index]} \n {df["sentence2"].iloc[index]}') for index in range(len(df['sentence1']))]
labels = df['label'].to_list()
guids = df['guid'].to_list()

train_y = df.iloc[:58114]['label']
valid_y = df.iloc[58114:61173]['label']

train_data_bow, valid_data_bow, test_data_bow = get_features(vectorizer="bow")
train_onehot_encoded, valid_onehot_encoded = get_one_hot_encode(train_y, valid_y)

model,preds = compute_model(train_data_bow, valid_data_bow, train_onehot_encoded, valid_onehot_encoded, model=1, patience=4)
compute_results(valid_y, preds)
plot_confusion_matrix(valid_y, preds, 'BoW + Neural Network', ['class 0', 'class 1', 'class 2', 'class 3'])
write_test_predictions(test_data_bow, model, "NeuralNetwork_Submission.csv")