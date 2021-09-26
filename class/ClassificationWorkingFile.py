# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 21:34:49 2021

@author: Slav
"""
# conda create --name py365 python=3.9.1 --channel conda-forge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# Train test split
from sklearn import model_selection

# Models
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from skopt import gp_minimize
from skopt.utils import cook_initial_point_generator
from skopt.space import Real, Integer
from skopt.plots import plot_evaluations, plot_objective, plot_convergence, plot_gaussian_process

# Read in data
train_set = pd.read_csv("./dataset/CS98XClassificationTrain.csv")
test_set = pd.read_csv("./dataset/CS98XClassificationTest.csv")

id_column = np.array(test_set["Id"])

# Drop all rows with NaNs
train_set = train_set.dropna()
# Drop all irrelevant columns
train_set = train_set.drop(columns=["Id", "title", "artist"])
test_set = test_set.drop(columns=["Id", "title", "artist"])
"""
train_set = train_set.groupby('top genre').filter(lambda x : len(x)>=5)
indeces = np.array(train_set[train_set['top genre'] == "album rock"].index)
indeces2 = np.array(train_set[train_set['top genre'] == "dance pop"].index)
indeces3 = np.array(train_set[train_set['top genre'] == "adult standards"].index)

indeces = indeces[:int(len(indeces)*0.45)]
indeces2 = indeces2[:int(len(indeces2)*0.45)]
indeces3 = indeces3[:int(len(indeces3)*0.45)]

train_set = train_set.drop(indeces.tolist(), axis=0)
train_set = train_set.drop(indeces2.tolist(), axis=0)
train_set = train_set.drop(indeces3.tolist(), axis=0)


for i in range(len(train_set.year.values)):
    train_set.year.values[i] = str(train_set.year.values[i])[2]

for i in range(len(test_set.year.values)):
    test_set.year.values[i] = str(test_set.year.values[i])[2]


columns = ["nrgy", "dnce", "dB", "val", "pop", "bpm", "live", "dur", "acous", "spch"]

for c in columns:
    train_set[c] = np.where((train_set[c] >= train_set[c].min()) & (train_set[c] <= train_set[c].quantile(0.25)) , 0, train_set[c])
    train_set[c] = np.where((train_set[c] > train_set[c].quantile(0.25)) & (train_set[c] <= train_set[c].quantile(0.50)) , 1, train_set[c])
    train_set[c] = np.where((train_set[c] > train_set[c].quantile(0.50)) & (train_set[c] <= train_set[c].quantile(0.75)) , 2, train_set[c])
    train_set[c] = np.where((train_set[c] > train_set[c].quantile(0.75)) & (train_set[c] <= train_set[c].max()) , 3, train_set[c])
"""

# What we want to predicttrain_set[c].int
predict = "top genre"

# Get everything except what we want to predict
X = np.array(train_set.drop([predict], 1)).astype(np.float64)
# Column we want to predict
y = np.array(train_set[predict])

X_scaled = scale(X)

# Train test split
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# --------------
# Validation Model Initialisation and Training
# --------------
print("> Support Vector Classifier")
classifier_svc = SVC(kernel="poly", degree=2, C=2, random_state=42)
classifier_svc.fit(X_train, y_train)

print("Accuracy score", classifier_svc.score(X_val, y_val))
y_pred = classifier_svc.predict(X_val)
#print(confusion_matrix(y_val, y_pred))
#print(classification_report(y_val, y_pred))

print("> Decission Tree Classifier")
classifier_dec_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
classifier_dec_tree.fit(X_train, y_train)

print("Accuracy score", classifier_dec_tree.score(X_val, y_val))
y_pred = classifier_dec_tree.predict(X_val)
#print(confusion_matrix(y_val, y_pred))
#print(classification_report(y_val, y_pred))

print("> Random Forest Classifier")
# IMPORTANT! n_jobs = -1 means it will use all available CPU cores. Change it if you need to
classifier_ran_tree = RandomForestClassifier(n_estimators=156, max_depth=28, min_samples_split=5,
                                                 min_samples_leaf=9, max_leaf_nodes=21,
                                                 bootstrap=True, n_jobs=-1, random_state=42)
classifier_ran_tree.fit(X_train, y_train)

print("Accuracy score", classifier_ran_tree.score(X_val, y_val))

print("> AdaBoost SAMME.R")
classifier_ada = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=4),
    n_estimators=500,
    algorithm="SAMME.R",
    learning_rate=0.2,
    random_state=42)

classifier_ada.fit(X_train, y_train)
print("Accuracy score", classifier_ada.score(X_val, y_val))

print("> Gradient Boost Classifier")
classifier_gb = GradientBoostingClassifier(n_estimators=100,
                                           learning_rate=1.0,
                                           max_depth=5,
                                           random_state=42)

classifier_gb.fit(X_train, y_train)
print("Accuracy score", classifier_gb.score(X_val, y_val))

print("> OVO Classifier")
classifier_ovo = OneVsOneClassifier(SVC(kernel="rbf", degree=3, C=1, random_state=42))
classifier_ovo.fit(X_train, y_train)
print("Accuracy score", classifier_ovo.score(X_val, y_val))

print("> OVR Classifier")
classifier_ovr = OneVsRestClassifier(SVC(kernel="rbf", degree=3, C=1, random_state=42))
classifier_ovr.fit(X_train, y_train)
print("Accuracy score", classifier_ovr.score(X_val, y_val))

print("> Voting Classifier Ensemble")
classifier_voting = VotingClassifier(
    estimators = [('sv', classifier_svc), ('rf', classifier_ran_tree), ('ovo', classifier_ovo)], voting = "hard")
classifier_voting.fit(X_train, y_train)
print("Accuracy score", classifier_voting.score(X_val, y_val))

print("> MLPC")
model2 = MLPClassifier(random_state=42)
model2.fit(X_train, y_train)
print("Accuracy score", model2.score(X_val, y_val))

print("> K-Nearest Neighbours")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("Accuracy score", model.score(X_val, y_val))





# --------------
# Testing Model Initialisation and Training
# --------------
X_test_forreal = np.array(test_set)
X_scaled_forreal = scale(X_test_forreal)
print("> Real Test Dataset")
# IMPORTANT! n_jobs = -1 means it will use all available CPU cores. Change it if you need to
#classifier_ran_tree = RandomForestClassifier(n_estimators=1000, max_depth=5, bootstrap=True, n_jobs=-1, random_state=42)
#classifier_ran_tree.fit(X, y)



real_classifier = RandomForestClassifier(n_estimators=133, max_depth=30, min_samples_split=3,
                                                 min_samples_leaf=12, max_leaf_nodes=15,
                                                 bootstrap=True, n_jobs=-1, random_state=42)
real_classifier.fit(X, y)

classifier_ovo = OneVsOneClassifier(SVC(kernel="poly", degree=3, C=1, random_state=42))
classifier_ovo.fit(X_train, y_train)

y_pred = classifier_ovo.predict(X_scaled_forreal)

csv = np.stack((id_column, y_pred), axis=1)
csv = np.vstack((np.array(["Id","top genre"]), csv))
np.savetxt("./foo.csv", csv, fmt='%s', delimiter=",")
print("> Optimisation")
def ada(x):
    classifier_ada = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=x[2]),
        n_estimators=x[0],
        algorithm="SAMME.R",
        learning_rate=x[1],
        random_state=42)

    classifier_ada.fit(X_train, y_train)
    return -classifier_ada.score(X_val, y_val)

def rf(x):
    classifier_ran_tree = RandomForestClassifier(n_estimators=x[0], max_depth=x[1], min_samples_split=x[2],
                                                 min_samples_leaf=x[3], max_leaf_nodes=x[4],
                                                 bootstrap=True, n_jobs=-1, random_state=42)
    classifier_ran_tree.fit(X_train, y_train)

    return -classifier_ran_tree.score(X_val, y_val)

def ovo(x):
    classifier_ovo = OneVsOneClassifier(SVC(kernel="rbf", degree=3, C=x[0], gamma="auto", random_state=42))
    classifier_ovo.fit(X_train, y_train)
    return -classifier_ovo.score(X_val, y_val)

lhs_maximin = cook_initial_point_generator("lhs", criterion="maximin")
# estimator, depth, minsamplesplit,
res = gp_minimize(ovo,
                      [Integer(1.0, 5.0, transform='identity', name='C')],
                      #xi=0.000001,
                      #kappa=0.001,
                      acq_func='gp_hedge', acq_optimizer='sampling',
                      n_calls=50, n_initial_points=14, initial_point_generator=lhs_maximin,
                      verbose = True,
                      noise = 1e-10,
                      random_state = 42)

_ = plot_evaluations(res)
plt.savefig('eval_result.png', bbox_inches='tight', dpi=300)

_ = plot_gaussian_process(res)
plt.savefig('gp.png', bbox_inches='tight', dpi=300)

_ = plot_objective(res, n_samples=50)
plt.savefig('objective.png', bbox_inches='tight', dpi=300)

_ = plot_convergence(res)
plt.savefig('convergence.png', bbox_inches='tight', dpi=300)

print(res.x, res.fun)
