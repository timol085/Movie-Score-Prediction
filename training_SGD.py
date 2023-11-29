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
import pickle

# read in dataset
reviews = pd.read_csv("IMDB Dataset.csv")

# take away all columns except ci and rc
reviews = reviews[['sentiment','review']]


# drop all empty reviews
reviews = reviews.dropna()

# read in the reviews and critic (rotten/fresh)
all_sent = reviews.sentiment
all_rev = reviews.review

# review from a specific movie
review_sent= all_sent[143:280]
review_rev = all_rev[143:280]

# shuffle data
all_sent, all_rev = shuffle(all_sent, all_rev, random_state=67)

# PIPELINE
my_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-7, random_state=42
,max_iter=500, tol=None)),
])

my_clf.fit(all_rev, all_sent)


# Save model and write to sav-file
filename = 'trainedModel_SGD.sav'
pickle.dump(my_clf,open(filename, 'wb'))
