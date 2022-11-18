import pandas as pd
import numpy as np
import os
from glob import iglob
from tabulate import tabulate
from joblib import load
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

feature = pd.read_pickle("./features/cycle_mean.pkl")
targets = []
for tar in iglob("./target_variables/*.pkl"):
    if tar == "./target_variables/full_set.pkl":
        continue
    print(tar)
    name = os.path.basename(tar)[:-4].split("_")
    name = " ".join(name).title()
    targets.append({'name':name, 'data': pd.read_pickle(tar)})

base_models = {}
for y_i in targets:
    X = feature
    y = y_i['data']
    y_dist = y.value_counts(normalize=True)

    scaler = StandardScaler()

    lr = LogisticRegression(
            max_iter = 400,
            verbose = 0,
            n_jobs = -1,
            random_state = 42
            )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    base_models.update({y_i['name']:{'model':lr.fit(X_train_scaled, y_train),
                           'predict':lr.predict(X_test_scaled),
                           'score': lr.score(X_test_scaled, y_test),
                           'train':(X_train_scaled, y_train),
                           'test': (X_test_scaled, y_test),
                           'y_dist': y_dist}})

for model in base_models.keys():
    print(f"\n{model}:")
    print(f"score: {base_models[model]['score']}\n")
    # print(f"{models[model]['y_dist']}")
