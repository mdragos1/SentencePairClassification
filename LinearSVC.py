import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, precision_score
import csv

def get_features(summed=False, vectorizer='tfidf'):
    """
    This function extracts features from sentences using TF-IDF
    or BoW. The 'summed' parameter makes it so the features are
    extracted from each sentence and then summed.
    """
    if(vectorizer == 'tf_idf'):
        vectorizer = TfidfVectorizer(
            max_features = None,
            strip_accents = 'unicode',
            analyzer = 'word',
            ngram_range = (1, 2))
    else:
        vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            analyzer='word',
            strip_accents = 'unicode',
            max_features=None)

    if(summed == False):
        train_data = vectorizer.fit_transform(sentences[:58114])
        valid_data = vectorizer.transform(sentences[58114:61173])
        test_data = vectorizer.transform(sentences[61173:])
        return train_data, valid_data, test_data

    s1_train_data = vectorizer.fit_transform(s1[:58114])
    s1_valid_data = vectorizer.transform(s1[58114:61173])
    s1_test_data = vectorizer.transform(s1[61173:])
    s2_train_data = vectorizer.transform(s2[:58114])
    s2_valid_data = vectorizer.transform(s2[58114:61173])
    s2_test_data = vectorizer.transform(s2[61173:])

    train_data = s1_train_data + s2_train_data
    valid_data = s1_valid_data + s2_valid_data
    test_data = s1_test_data + s2_test_data
    return train_data, valid_data, test_data


def plot_confusion_matrix(y_true, predicted, title, labels=None, normalize=None):
    """
    This function plots a confusion matrix with 6x6 size and a 100 dpi resolution.
    """

    #I changed the figure size and increased the dpi for a better resolution.
    _, axes = plt.subplots(figsize=(6,6), dpi=100)

    conf_matrix = confusion_matrix(y_true, predicted, normalize=normalize)
    display = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)

    #I set the title using the axes object.
    axes.set(title=f'Confusion Matrix {title} Model')

    display.plot(ax=axes)
    return conf_matrix

def compute_model(train_data, train_y, valid_data, C=10, max_iter=5000, loss='hinge'): 
    """
    This function creates a new LinearSVC model with the specified parameters,
    fits the model, makes prediction for the validation data and returns the
    the model and predictions.
    """
    model = LinearSVC(C=C, max_iter=max_iter, loss=loss)
    model.fit(train_data, train_y)
    preds = model.predict(valid_data)
    return model, preds

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

train_data_bow, valid_data_bow, test_data_bow = get_features(summed=True, vectorizer="bow")

train_y = df.iloc[:58114]['label']
valid_y = df.iloc[58114:61173]['label']

model,preds = compute_model(train_data_bow, train_y, valid_data_bow, C=0.1)
compute_results(valid_y, preds)
plot_confusion_matrix(valid_y, preds, 'BoW + LinearSVC', ['class 0', 'class 1', 'class 2', 'class 3'])
write_test_predictions(test_data_bow, model, "LinearSVC_Submision.csv")