from typing import Union, BinaryIO
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

class RegularModel:

    def __init__(self, X_path:Union[BinaryIO, str], y_path:Union[BinaryIO, str], X_name:str='', y_name:str='', random_state:int=42, test_size:float=0.15):
        self._X = pd.read_pickle(X_path)
        self._y = pd.read_pickle(y_path)
        self.features = X_name
        self.target = y_name
        self.rs = random_state
        self.test_size = test_size
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, 
                                                                                        self._y, 
                                                                                        test_size = self.test_size, 
                                                                                        random_state = self.rs)

        _knn_pipe = Pipeline([('scaler', StandardScaler()),
                            ('knn', KNeighborsClassifier())])
        
        _xgb_pipe = Pipeline([('scaler', StandardScaler()),
                             ('xgb', XGBClassifier(random_state = self.rs))])

        param_range = [1, 2, 3, 4, 5, 6]
        param_range_fl = [1.0, 0.5, 0.1]
        n_estimators = [50, 100, 150]
        learning_rates = [.1, .2, .3]

        knn_param_grid = [{'knn__n_neighbors': param_range,
                           'knn__weights': ['uniform', 'distance'],
                           'knn__metric': ['euclidean', 'manhattan']}]

        xgb_param_grid = [{'xgb__learning_rate': learning_rates,
                            'xgb__max_depth': param_range,
                            'xgb__min_child_weight': param_range[:2],
                            'xgb__subsample': param_range_fl,
                            'xgb__n_estimators': n_estimators}]

        self.knn_grid_search = GridSearchCV(estimator = _knn_pipe,
                                       param_grid = knn_param_grid,
                                       scoring = 'accuracy',
                                       cv = 3,
                                       n_jobs = -1)

        self.xgb_grid_search = GridSearchCV(estimator = _xgb_pipe,
                                       param_grid = xgb_param_grid,
                                       scoring = 'accuracy',
                                       cv = 3,
                                       n_jobs = -1)
        
        self.grids = [self.knn_grid_search, self.xgb_grid_search]

    def fit_model(self):
        for i in self.grids:
            i.fit(self._X_train, self._y_train)
        return self
    
    def save_model(self, filename:Union[BinaryIO, str]):
        if not 'dump' in dir():
            from joblib import dump
        dump(self, filename)

    def report(self, how:str="print", where:Union[BinaryIO, str]=""):
        grid_dict = {0: 'K-Nearest Neighbors', 1: 'XGBoost'}
        rep_list = []
        rep_list.append(f"Feature type: {self.features}\nTarget Variable: {self.target}\n")
        rep_list.append('\n')
        for i, model in enumerate(self.grids):
            rep_list.append('{} Test Accuracy: {}\n'.format(grid_dict[i],\
            model.score(self._X_test, self._y_test)))
            rep_list.append('{} Best Params: {}\n'.format(grid_dict[i], model.best_params_))
            rep_list.append('\n')
        if how == "file":
            if not where:
                raise ValueError("You must pass a string with filename and path to use 'file' output method.")
            with open(where, 'w') as file:
                file.writelines(rep_list)
                return print(f"Report ouput to {where}.")
        return print(*rep_list)
