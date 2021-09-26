# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:01:09 2021

@author: Slav
"""
# Essential Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.preprocessing import StandardScaler, scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# Model Selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# ML Models
from sklearn.svm import SVC, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Metrics
from sklearn.metrics import mean_squared_error

# Scikit-Optimise
from skopt import gp_minimize
from skopt.utils import cook_initial_point_generator
from skopt.space import Real, Integer

train_set = pd.read_csv("./dataset/CS98XRegressionTrain.csv")
test_set = pd.read_csv("./dataset/CS98XRegressionTest.csv")
train_set = train_set.rename({'pop': 'popularity'}, axis='columns')
id_column = np.array(test_set["Id"])

#for i in range(len(train_set.year.values)):
#    train_set.year.values[i] = str(train_set.year.values[i])[2]

#for i in range(len(test_set.year.values)):
#    test_set.year.values[i] = str(test_set.year.values[i])[2]

# Drop all rows with NaNs
train_set = train_set.dropna()
if(test_set["top genre"].isnull().values.any()):
	test_set = test_set.replace(np.nan, "other")
#print(test_set["top genre"].iloc[66])
"""
# JOIN GENRES TOGETHER
genre_array = np.array([["rock", "rock"], ["invasion", "rock"],
						 ["pop", "pop"], ["r&b", "pop"],
						 ["jazz", "jazz"],
						 ["blues", "blues"],
						 ["metal", "metal"],
						 ["folk", "folk"],
						 ["hip hop", "hip hop"], ["rap", "hip hop"],
						 ["funk", "funk"],
						 ["soul", "soul"],
						 ["adult standards", "adult standards"],
						 ["dance", "dance"], ["big room", "dance"], ["house", "dance"], ["hi-nrg", "dance"]])

for pair in genre_array:
	train_set.loc[train_set["top genre"].str.contains(pair[0]), "top genre"] = pair[1]

for pair in genre_array:
	test_set.loc[test_set["top genre"].str.contains(pair[0]), "top genre"] = pair[1]

#train_set = train_set.groupby("top genre").filter(lambda x : len(x)>=6)
#test_set = test_set.groupby("top genre").filter(lambda x : len(x)>=6)

# ONE HOT ENC ON GENRES
def onehot(t):
	encoder = OneHotEncoder()
	genre_encoded, genre_categories = t["top genre"].factorize()
	genre_1hot = encoder.fit_transform(genre_encoded.reshape(-1,1))
	enc_data = pd.DataFrame(genre_1hot.toarray())
	enc_data.columns = genre_categories
	enc_data.index = t.index
	t = t.join(enc_data)
	return t

train_set = onehot(train_set)
test_set = onehot(test_set)
"""

# Drop all irrelevant columns, maybe also "live", "dur", "top genre"
train_set = train_set.drop(columns=["Id", "title", "artist", "top genre"])
test_set = test_set.drop(columns=["Id", "title", "artist", "top genre"])

"""
# differences between columns in data sets
train_diff = train_set[train_set.columns.difference(test_set.columns)]

for col in train_diff.columns:
	if(not (col == "popularity")):
		test_set[col] = np.zeros(test_set.shape[0])

test_diff = test_set[test_set.columns.difference(train_set.columns)]
test_set = test_set.drop(columns=test_set[test_diff.columns])
"""

# What we want to predict
predict = "popularity"

# Get everything except what we want to predict
X = np.array(train_set.drop([predict], 1)).astype(np.float64)
# Column we want to predict
y = np.array(train_set[predict])
X_test = np.array(test_set).astype(np.float64)

#pca = PCA(n_components=1)
#X = pca.fit_transform(X)

std_scaler = StandardScaler().fit(X)
X_scaled = std_scaler.transform(X)
X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Non-scaled data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# --------------
# Model Testing
# --------------

models = []

def rnd_for():
    model = RandomForestRegressor(n_estimators=1000, max_depth=25,
                                  bootstrap=True, n_jobs=-1,
                                  warm_start=True, random_state=42)
    model.fit(X_train, y_train)
    models.append(("rnd_for", model))
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print("> Random Forest Regressor", rmse)


def knnr():
    model = KNeighborsRegressor(n_neighbors=9,
                                 n_jobs=-1)
    model.fit(X_train, y_train)
    models.append(("knnc", model))
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print("> K-Nearest Neighbors Regressor", rmse)

def mlpr():
    model = MLPRegressor(activation = "logistic", solver="adam",
                          max_iter=10000, alpha=0.1,
                          learning_rate_init=0.05, warm_start=True,
                          learning_rate="invscaling", shuffle=False,
                          random_state=42)
    model.fit(X_train_scaled, y_train)
    models.append(("mlpr", model))
    y_pred = model.predict(X_val_scaled)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print("> Multi-Layer Perceptron Regressor", rmse)

print("Using columns", train_set.columns.values[:len(train_set.columns.values)-1])
print("---Evaluating through rmse; lower is better---")


rnd_for()
knnr()
mlpr()

# ----------------
# Real Prediction
# ----------------
def real():
	model = RandomForestRegressor(n_estimators=1000, max_depth=25,
                                  bootstrap=True, n_jobs=-1,
                                  warm_start=True, random_state=42)
	model.fit(scale(X), y)
	y_pred = model.predict(scale(X_test))
	csv = np.stack((id_column.astype(np.int32), y_pred.astype(np.int32)), axis=1)
	csv = np.vstack((np.array(["Id","pop"]), csv))
	np.savetxt("./foo.csv", csv, fmt='%s', delimiter=",")

real()


# --------------
# Optimisation
# --------------

def f(x):
    model = RandomForestRegressor(n_estimators=x[0], max_depth=x[1],
                                  bootstrap=True, n_jobs=-1,
                                  warm_start=True, random_state=42)
    model.fit(X_train, y_train)
    models.append(("rnd_for", model))
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    return rmse

def bayesian_opt():
    lhs_maximin = cook_initial_point_generator("lhs", criterion="maximin")

    res = gp_minimize(f,
                      [Integer(2, 1000, transform='identity'),
                       Integer(2, 100, transform='identity')],
                      #xi=0.000001,
                      #kappa=0.001,
                      acq_func='EI', acq_optimizer='sampling',
                      n_calls=100, n_initial_points=10, initial_point_generator=lhs_maximin,
                      verbose = True,
                      noise = 1e-10,
                      random_state = 42)

    return (res.x, res.fun)

#print(bayesian_opt())