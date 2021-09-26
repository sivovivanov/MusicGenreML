# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 14:54:42 2021

@author: Slav
"""
# Essential Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.preprocessing import StandardScaler, scale
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Model Selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# ML Models
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Metrics
from sklearn.metrics import f1_score

# Scikit-Optimise
from skopt import gp_minimize
from skopt.utils import cook_initial_point_generator
from skopt.space import Real, Integer

# Read in data
train_set = pd.read_csv("./dataset/CS98XClassificationTrain.csv")
test_set = pd.read_csv("./dataset/CS98XClassificationTest.csv")

id_column = np.array(test_set["Id"])

# Drop all rows with NaNs
train_set = train_set.dropna()
# Drop all irrelevant columns "Id", "title", "artist", "spch", "nrgy", "live", "dB", "bpm"
train_set = train_set.drop(columns=["Id", "title", "artist", "spch", "nrgy", "live", "dB", "bpm"])
test_set = test_set.drop(columns=["Id", "title", "artist", "spch", "nrgy", "live", "dB", "bpm"])
train_set = train_set.groupby('top genre').filter(lambda x : len(x)>=6)
"""
for i in range(len(train_set.year.values)):
    train_set.year.values[i] = str(train_set.year.values[i])[2]

for i in range(len(test_set.year.values)):
    test_set.year.values[i] = str(test_set.year.values[i])[2]
	
def onehot(t, column):
	encoder = OneHotEncoder()
	column_encoded, column_categories = t[column].factorize()
	column_1hot = encoder.fit_transform(column_encoded.reshape(-1,1))
	enc_data = pd.DataFrame(column_1hot.toarray())
	enc_data.columns = column_categories
	enc_data.index = t.index
	t = t.join(enc_data)
	return t

train_set = onehot(train_set, "year")
test_set = onehot(test_set, "year")

# differences between columns in data sets
train_diff = train_set[train_set.columns.difference(test_set.columns)]

for col in train_diff.columns:
	if(not (col == "top genre")):
		test_set[col] = np.zeros(test_set.shape[0])

test_diff = test_set[test_set.columns.difference(train_set.columns)]
test_set = test_set.drop(columns=test_set[test_diff.columns])
"""

# What we want to predict
predict = "top genre"
#train_set = train_set.groupby(predict).filter(lambda x : len(x)>=6)

# Get everything except what we want to predict
X = np.array(train_set.drop([predict], 1)).astype(np.float64)
# Column we want to predict
y = np.array(train_set[predict])

"""
oversample = RandomOverSampler(sampling_strategy='auto')
undersample = RandomUnderSampler(sampling_strategy='auto')
X, y = oversample.fit_resample(X, y)
X, y = undersample.fit_resample(X, y)
"""
# Actual test dataset as numpy array
X_test = np.array(test_set).astype(np.float64)

# Scaled data
std_scaler = StandardScaler()
X_scaled = std_scaler.fit_transform(X)
X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Non-scaled data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# --------------
# Model Testing
# --------------

models = []

def ovo():
    model = OneVsOneClassifier(SVC(kernel="rbf", degree=2, C=1.1, gamma=0.3,
                                   probability=True, decision_function_shape = "ovo",
                                   random_state=42))
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    models.append(("ovo", model))
    print("> One Versus One Classifier", model.score(X_val_scaled, y_val),
          "- f1", f1_score(y_val, y_pred, average="weighted"))

def ovr():
    model = OneVsRestClassifier(LinearSVC(dual=False, fit_intercept=False,
                                          max_iter=10000,
                                          random_state=42))
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    models.append(("ovr", model))
    print("> One Versus Rest Classifier", model.score(X_val_scaled, y_val),
          "- f1", f1_score(y_val, y_pred, average="weighted"))

def dec_tree():
    model = DecisionTreeClassifier(max_depth=2, splitter="best",
                                   max_features="auto", criterion="gini",
                                   random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    models.append(("dec_tree", model))
    print("> Decision Tree Classifier", model.score(X_val, y_val),
          "- f1", f1_score(y_val, y_pred, average="weighted"))

def rnd_for():
    model = RandomForestClassifier(n_estimators=155, max_depth=2,
                                                 bootstrap=True, n_jobs=-1,
                                                 warm_start=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    models.append(("rnd_for", model))
    print("> Random Forest Classifier", model.score(X_val, y_val),
          "- f1", f1_score(y_val, y_pred, average="weighted"))

def ada():
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, max_features="auto",),
                               n_estimators=40,
                               algorithm="SAMME.R",
                               learning_rate=0.2,
                               random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    models.append(("ada", model))
    print("> AdaBoost Classifier", model.score(X_val, y_val),
          "- f1", f1_score(y_val, y_pred, average="weighted"))

def gbc():
    model = GradientBoostingClassifier(max_depth=4, n_estimators=50,
                                       learning_rate=1.0, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    models.append(("gbc", model))
    print("> Gradient Boosting Classifier", model.score(X_val, y_val),
          "- f1", f1_score(y_val, y_pred, average="weighted"))
def mlpc():
    model = MLPClassifier(activation = "logistic", solver="adam",
                          max_iter=10000, alpha=0.1,
                          learning_rate_init=0.05, warm_start=True,
                          learning_rate="invscaling", shuffle=False,
                          random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    models.append(("mlpc", model))
    print("> Multi-Layer Perceptron Classifier", model.score(X_val_scaled, y_val),
          "- f1", f1_score(y_val, y_pred, average="weighted"))

def knnc():
    model = KNeighborsClassifier(n_neighbors=15, weights="uniform",
                                 n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    models.append(("knnc", model))
    print("> K-Nearest Neighbors Classifier", model.score(X_val, y_val),
          "- f1", f1_score(y_val, y_pred, average="weighted"))

def voting():
    model = VotingClassifier(estimators=models,
        voting='hard')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    print("> Voting Classifier", model.score(X_val_scaled, y_val),
          "- f1", f1_score(y_val, y_pred, average="weighted"))

print("Using columns", train_set.columns.values[:len(train_set.columns.values)-1])
ovo()
ovr()
dec_tree()
rnd_for()
ada()
mlpc()
knnc()
voting()

# --------------
# Real Prediction
# --------------

model = OneVsRestClassifier(LinearSVC(dual=False, fit_intercept=False,
                                          max_iter=10000,
                                          random_state=42))
model.fit(X, y)
y_pred = model.predict(X_test)

csv = np.stack((id_column, y_pred), axis=1)
csv = np.vstack((np.array(["Id","top genre"]), csv))
np.savetxt("./foo.csv", csv, fmt='%s', delimiter=",")

# --------------
# Optimisation
# --------------

def f(x):
    model = RandomForestClassifier(n_estimators=155, max_depth=2,
                                   bootstrap=True, n_jobs=-1,
                                   warm_start=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    return -model.score(X_val_scaled, y_val)

def bayesian_opt():
    lhs_maximin = cook_initial_point_generator("lhs", criterion="maximin")

    res = gp_minimize(f,
                      [Integer(1, 100, transform='identity'),
                       Integer(10, 1000, transform='identity'),
                       Real(0.1, 100.0, transform='identity')],
                      x0=[[3, 40, 0.2]],
                      xi=0.000001,
                      #kappa=0.001,
                      acq_func='EI', acq_optimizer='sampling',
                      n_calls=100, n_initial_points=10, initial_point_generator=lhs_maximin,
                      verbose = False,
                      noise = 1e-10,
                      random_state = 42)

    return (res.x, res.fun)
