import numpy as np
import pandas as pd
from pandas.errors import ParserError
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, plot_roc_curve


class RegularModel:

    def __init__(self, X, y, X_name:str='', y_name:str=''):
        self.model1_X = X
        self.model1_y = y
        self.features = X_name
        self.target = y_name

        self.model1_X_train, self.model1_X_test, self.model1_y_train, self.model1_y_test = train_test_split(self.model1_X, 
                                                                                        self.model1_y, 
                                                                                        test_size = 0.15, 
                                                                                        random_state = 42)

        model1_lr_pipe = Pipeline([('scaler', StandardScaler()),
                            ('lr', LogisticRegression(random_state = 42))])

        model1_dtree_pipe = Pipeline([('scaler', StandardScaler()),
                            ('dtree',DecisionTreeClassifier(random_state = 42))])

        model1_rf_pipe = Pipeline([('scaler', StandardScaler()),
                            ('rf',RandomForestClassifier(random_state = 42))])

        model1_knn_pipe = Pipeline([('scaler', StandardScaler()),
                            ('knn', KNeighborsClassifier())])

        model1_svm_pipe = Pipeline([('scaler', StandardScaler()),
                             ('svm', svm.SVC(random_state = 42))])

        model1_xgb_pipe = Pipeline([('scaler', StandardScaler()),
                             ('xgb', XGBClassifier(random_state = 42))])

        param_range = [1, 2, 3, 4, 5, 6]
        param_range_fl = [1.0, 0.5, 0.1]
        n_estimators = [50, 100, 150]
        learning_rates = [.1, .2, .3]

        lr_param_grid = [{'lr__penalty': ['l1', 'l2'],
                           'lr__C': param_range_fl,
                           'lr__solver': ['liblinear']}]

        dtree_param_grid = [{'dtree__criterion': ['gini', 'entropy'],
                           'dtree__min_samples_leaf': param_range,
                           'dtree__max_depth': param_range,
                           'dtree__min_samples_split': param_range[1:]}]

        rf_param_grid = [{'rf__min_samples_leaf': param_range,
                           'rf__max_depth': param_range,
                           'rf__min_samples_split': param_range[1:]}]

        knn_param_grid = [{'knn__n_neighbors': param_range,
                           'knn__weights': ['uniform', 'distance'],
                           'knn__metric': ['euclidean', 'manhattan']}]

        svm_param_grid = [{'svm__kernel': ['linear', 'rbf'], 
                            'svm__C': param_range}]

        xgb_param_grid = [{'xgb__learning_rate': learning_rates,
                            'xgb__max_depth': param_range,
                            'xgb__min_child_weight': param_range[:2],
                            'xgb__subsample': param_range_fl,
                            'xgb__n_estimators': n_estimators}]

        lr_grid_search = GridSearchCV(estimator = model1_lr_pipe,
                                      param_grid = lr_param_grid,
                                      scoring = 'accuracy',
                                      cv = 3,
                                      n_jobs = -1)

        dtree_grid_search = GridSearchCV(estimator = model1_dtree_pipe,
                                         param_grid = dtree_param_grid,
                                         scoring = 'accuracy',
                                         cv = 3,
                                         n_jobs = -1)

        rf_grid_search = GridSearchCV(estimator = model1_rf_pipe,
                                      param_grid = rf_param_grid,
                                      scoring = 'accuracy',
                                      cv = 3,
                                      n_jobs = -1)

        knn_grid_search = GridSearchCV(estimator = model1_knn_pipe,
                                       param_grid = knn_param_grid,
                                       scoring = 'accuracy',
                                       cv = 3,
                                       n_jobs = -1)

        svm_grid_search = GridSearchCV(estimator = model1_svm_pipe,
                                       param_grid = svm_param_grid,
                                       scoring = 'accuracy',
                                       cv = 3,
                                       n_jobs = -1)

        xgb_grid_search = GridSearchCV(estimator = model1_xgb_pipe,
                                       param_grid = xgb_param_grid,
                                       scoring = 'accuracy',
                                       cv = 3,
                                       n_jobs = -1)


        self.grids = [lr_grid_search, dtree_grid_search, rf_grid_search, knn_grid_search, svm_grid_search, xgb_grid_search]

    def fit_model(self):
        for i in self.grids:
            i.fit(self.model1_X_train, self.model1_y_train)
        return self

    def print_m(self):
        grid_dict = {0: 'Logistic Regression', 1: 'Decision Trees', 
                     2: 'Random Forest', 3: 'K-Nearest Neighbors', 
                     4: 'Support Vector Machines', 5: 'XGBoost'}
        
        print(f"Feature type: {self.features}\nTarget Variable: {self.target}")
        for i, model in enumerate(self.grids):
            print('{} Test Accuracy: {}'.format(grid_dict[i],\
            model.score(self.model1_X_test, self.model1_y_test)))
            print('{} Best Params: {}'.format(grid_dict[i], model.best_params_))
            print('\v')
