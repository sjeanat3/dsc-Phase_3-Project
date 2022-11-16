import glob
import os
import pickle
from typing import BinaryIO, Union, Iterable, Tuple, Callable
from joblib import dump, load

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, plot_confusion_matrix,
                             plot_roc_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class MultiModel:

    def __init__(self, X_path:Union[BinaryIO, str], y_path:Union[BinaryIO, str], X_name:str='', y_name:str=''):
        self.model_X = pd.read_pickle(X_path)
        self.model_y = pd.read_pickle(y_path)
        self.features = X_name
        self.target = y_name

        self.model_X_train, self.model_X_test, self.model_y_train, self.model_y_test = train_test_split(self.model_X, 
                                                                                        self.model_y, 
                                                                                        test_size = 0.15, 
                                                                                        random_state = 42)

        model_lr_pipe = Pipeline([('scaler', StandardScaler()),
                            ('lr', LogisticRegression(random_state = 42))])

        model_dtree_pipe = Pipeline([('scaler', StandardScaler()),
                            ('dtree',DecisionTreeClassifier(random_state = 42))])

        model_rf_pipe = Pipeline([('scaler', StandardScaler()),
                            ('rf',RandomForestClassifier(random_state = 42))])

        model_knn_pipe = Pipeline([('scaler', StandardScaler()),
                            ('knn', KNeighborsClassifier())])

        model_svm_pipe = Pipeline([('scaler', StandardScaler()),
                             ('svm', svm.SVC(random_state = 42))])

        model_xgb_pipe = Pipeline([('scaler', StandardScaler()),
                             ('xgb', XGBClassifier(random_state = 42))])

        self.param_range = [1, 2, 3, 4, 5, 6]
        self.param_range_fl = [1.0, 0.5, 0.1]
        self.n_estimators = [50, 100, 150]
        self.learning_rates = [.1, .2, .3]

        lr_param_grid = [{'lr__penalty': ['l1', 'l2'],
                           'lr__C': self.param_range_fl,
                           'lr__solver': ['liblinear']}]

        dtree_param_grid = [{'dtree__criterion': ['gini', 'entropy'],
                           'dtree__min_samples_leaf': self.param_range,
                           'dtree__max_depth': self.param_range,
                           'dtree__min_samples_split': self.param_range[1:]}]

        rf_param_grid = [{'rf__min_samples_leaf': self.param_range,
                           'rf__max_depth': self.param_range,
                           'rf__min_samples_split': self.param_range[1:]}]

        knn_param_grid = [{'knn__n_neighbors': self.param_range,
                           'knn__weights': ['uniform', 'distance'],
                           'knn__metric': ['euclidean', 'manhattan']}]

        svm_param_grid = [{'svm__kernel': ['linear', 'rbf'], 
                            'svm__C': self.param_range}]

        xgb_param_grid = [{'xgb__learning_rate': self.learning_rates,
                            'xgb__max_depth': self.param_range,
                            'xgb__min_child_weight': self.param_range[:2],
                            'xgb__subsample': self.param_range_fl,
                            'xgb__n_estimators': self.n_estimators}]

        lr_grid_search = GridSearchCV(estimator = model_lr_pipe,
                                      param_grid = lr_param_grid,
                                      scoring = 'accuracy',
                                      cv = 3,
                                      n_jobs = -1)

        dtree_grid_search = GridSearchCV(estimator = model_dtree_pipe,
                                         param_grid = dtree_param_grid,
                                         scoring = 'accuracy',
                                         cv = 3,
                                         n_jobs = -1)

        rf_grid_search = GridSearchCV(estimator = model_rf_pipe,
                                      param_grid = rf_param_grid,
                                      scoring = 'accuracy',
                                      cv = 3,
                                      n_jobs = -1)

        knn_grid_search = GridSearchCV(estimator = model_knn_pipe,
                                       param_grid = knn_param_grid,
                                       scoring = 'accuracy',
                                       cv = 3,
                                       n_jobs = -1)

        svm_grid_search = GridSearchCV(estimator = model_svm_pipe,
                                       param_grid = svm_param_grid,
                                       scoring = 'accuracy',
                                       cv = 3,
                                       n_jobs = -1)

        xgb_grid_search = GridSearchCV(estimator = model_xgb_pipe,
                                       param_grid = xgb_param_grid,
                                       scoring = 'accuracy',
                                       cv = 3,
                                       n_jobs = -1)


        self.grids = {"lr_grid_search":lr_grid_search, "dtree_grid_search":dtree_grid_search, "rf_grid_search":rf_grid_search, "knn_grid_search":knn_grid_search, "svm_grid_search":svm_grid_search, "xgb_grid_search":xgb_grid_search}

    def fit_model(self):
        for i in self.grids.values():
            i.fit(self.model_X_train, self.model_y_train)
        return self

    def print_m(self):
        file_lines = []
        grid_dict = {0: 'Logistic Regression', 1: 'Decision Trees', 
                     2: 'Random Forest', 3: 'K-Nearest Neighbors', 
                     4: 'Support Vector Machines', 5: 'XGBoost'}
        
        file_lines.append("Model_Output\n")
        file_lines.append("---------------------------------------------------------------\n")
        file_lines.append(f"Feature type: {self.features}\nTarget Variable: {self.target}\n")
        for i, model in enumerate(self.grids.values()):
            file_lines.append("{} Test Accuracy: {}\n".format(grid_dict[i], model.score(self.model_X_test, self.model_y_test)))
            file_lines.append('{} Best Params: {}\n'.format(grid_dict[i], model.best_params_))
            file_lines.append('\n')
        file_lines.append("end of file...\n")
        return file_lines
