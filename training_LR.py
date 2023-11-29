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
reviews = pd.read_csv("./input/IMDB Dataset.csv")

# take away all columns except ci and rc
reviews = reviews[["sentiment", "review"]]


# drop all empty reviews
reviews = reviews.dropna()

# read in the reviews and critic (rotten/fresh)
all_ci = reviews.sentiment
all_rc = reviews.review

# review from a specific movie
review_ci = all_ci[143:280]
review_rc = all_rc[143:280]

# shuffle data
all_ci, all_rc = shuffle(all_ci, all_rc, random_state=67)

# take out shuffled training set with 100000
train_ci = all_ci[:199999]
train_rc = all_rc[:199999]

test_ci = all_ci[100000:199999]
test_rc = all_rc[100000:199999]

# PIPELINE
tomato_clf = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", LogisticRegression(solver="liblinear", multi_class="auto")),
    ]
)
# ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-5, random_state=42
# ,max_iter=8, tol=None)),
# ])
# ('clf', MultinomialNB()),
# ])

tomato_clf.fit(all_rc, all_ci)
# print("hej")

filename = "trainedModel_LR.sav"
pickle.dump(tomato_clf, open(filename, "wb"))
