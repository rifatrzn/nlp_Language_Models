import csv
from importlib import reload
import nltk
from nltk.lm import MLE
from nltk.util import ngrams
from nltk.lm import Vocabulary, MLE, Laplace
import sys


# Load the data file from the path
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = list(csv.reader(file))
    return data


# function to preprocess text data
# convert to lowercase
# remove newline characters
def preprocess(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    return text


# function to train LM on train data
# Word tokenization and bigrams and ngrams
def train_LM(train_data):
    train_text = [preprocess(row[1]) for row in train_data]
    train_tokens = [nltk.word_tokenize(text) for text in train_text]
    train_bigrams = [ngrams(tokens, 2) for tokens in train_tokens]

    lm = MLE(2)
    lm.vocab = nltk.lm.Vocabulary([word for sent in train_data for word in sent], unk_cutoff=2)
    lm.fit(train_bigrams)

    return lm


# function to test LM on data
def test_LM(data, lm, output_path):
    test_text = [preprocess(row[1]) for row in data]
    test_tokens = [nltk.word_tokenize(text) for text in test_text]
    scores = []
    for tokens in test_tokens:
        bigrams = ngrams(tokens, 2)
        score = lm.logscore(bigrams)
        scores.append(score)

    # write output to csv file
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'comment_text', 'score'])
        for i, row in enumerate(data):
            writer.writerow([row[0], row[1], scores[i]])


# loading data from file path
train_data = load_data('E:/NLP Project/rifat_jaham_rezwan_hw03/train.csv')

test_data = load_data('E:/NLP Project/rifat_jaham_rezwan_hw03/test.csv')

test_labels = load_data('E:/NLP Project/rifat_jaham_rezwan_hw03/test_labels.csv')

# creating Language Models
LM_full = train_LM(train_data)
LM_not = train_LM([row for row in train_data if row[2] == '0'])
LM_toxic = train_LM([row for row in train_data if row[2] == '1'])

# test Language models on test data
test_full = test_data
test_not = [row for row in test_data if test_labels[test_data.index(row)][1:] == ['0', '0', '0', '0', '0', '0']]
test_toxic = [row for row in test_data if test_labels[test_data.index(row)][1:] != ['0', '0', '0', '0', '0', '0']]

# output cretes new csv file and data
test_LM(test_full, LM_full, 'LM_full_test_results.csv')
test_LM(test_not, LM_full, 'LM_full_not_test_results.csv')
test_LM(test_toxic, LM_full, 'LM_full_toxic_test_results.csv')

test_LM(test_full, LM_not, 'LM_not_test_results.csv')
test_LM(test_not, LM_not, 'LM_not_not_test_results.csv')
test_LM(test_toxic, LM_not, 'LM_not_toxic_test_results.csv')

test_LM(test_full, LM_toxic, 'LM_toxic_test_results.csv')
test_LM(test_not, LM_toxic, 'LM_toxic_not_test_results.csv')
test_LM(test_toxic, LM_toxic, 'LM_toxic_toxic_test_results.csv')
