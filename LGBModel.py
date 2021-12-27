import lightgbm as lgb
from sklearn.metrics import accuracy_score
import optuna
import numpy as np
import pandas as pd


class LGBModel:
    def __init__(self, x, y, num_folds, fold_feature, params=None, models=None):
        self.x = x
        self.y = y
        self.num_folds = num_folds
        self.fold_feature = fold_feature
        self.params = params
        self.models = models

    def train(self, params):
        self.models = [None] * self.num_folds

        full_preds = np.zeros(len(self.y))

        for fold in range(self.num_folds):
            x_train = self.x.loc[self.x[self.fold_feature] != fold].drop(self.fold_feature, axis=1)
            y_train = self.y.loc[self.y[self.fold_feature] != fold].drop(self.fold_feature, axis=1)
            x_valid = self.x.loc[self.x[self.fold_feature] == fold].drop(self.fold_feature, axis=1)
            y_valid = self.y.loc[self.y[self.fold_feature] == fold].drop(self.fold_feature, axis=1)
            dtrain = lgb.Dataset(x_train, label=y_train)
            self.models[fold] = lgb.train(params, dtrain)

            full_preds[y_valid.index] = self.models[fold].predict(x_valid)

        return full_preds

    def objective(self, trial):

        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

        preds = self.train(param)

        return accuracy_score(self.y.drop(self.fold_feature, axis=1), np.rint(preds))

    def fit(self, num_trials=100):
        if self.params is None:
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=num_trials)
            self.params = study.best_trial.params

        preds = self.train(self.params)
        print(f'LGBScore: {accuracy_score(self.y.drop(self.fold_feature, axis=1), np.rint(preds))}')

        df = pd.DataFrame(preds)
        df.columns = ['lgb']

        return df

    def predict(self, x):
        preds = pd.DataFrame()
        preds.index = range(len(x))
        preds['lgb'] = 0.0
        for i in range(self.num_folds):
            preds['lgb'] = preds['lgb'] + pd.Series(self.models[i].predict(x))

        preds['lgb'] = preds['lgb'] / float(self.num_folds)

        return preds

