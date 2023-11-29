from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
import sklearn
from sklearn.datasets import load_files
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle

# read in dataset
reviews = pd.read_csv("IMDB Dataset.csv")

movie_reviews = pd.read_csv("frozen2.csv")
movie_reviews.drop_duplicates(subset ="review", keep = 'first', inplace = True)

# take away all columns except ci and rc
reviews = reviews[['sentiment','review']]


# drop all empty reviews
reviews = reviews.dropna()

# read in the reviews and critic (rotten/fresh)
all_sent = reviews.sentiment
all_rev = reviews.review

# shuffle data
all_sent, all_rev = shuffle(all_sent, all_rev, random_state=67)


# test data
test_sent = all_sent[100000:199999]
test_rev = all_rev[100000:199999]

# tomatometer model
predictionModel = pickle.load(open('trainedModel_SGD.sav', 'rb'))


predicted = predictionModel.predict(movie_reviews.review)
print(len(movie_reviews))
print(len(predicted))
#print("confusion matrix: ", metrics.confusion_matrix(movie_reviews, predicted))

print("Predicted score by ussss: ", np.mean(predicted == 'positive'))

# GRID SEARCH
