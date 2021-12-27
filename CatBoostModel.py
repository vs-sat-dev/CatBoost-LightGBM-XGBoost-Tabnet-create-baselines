from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import optuna
import numpy as np
import pandas as pd


class CatBoostModel:
    def __init__(self, x, y, num_folds, fold_feature, params=None, models=None):
        self.x = x
        self.y = y
        self.num_folds = num_folds
        self.fold_feature = fold_feature
        self.params = params
        self.models = models

    def train(self, params):
        self.models = [CatBoostClassifier(**params) for _ in range(self.num_folds)]

        full_preds = np.zeros(len(self.y))

        for fold in range(self.num_folds):
            x_train = self.x.loc[self.x[self.fold_feature] != fold].drop(self.fold_feature, axis=1)
            y_train = self.y.loc[self.y[self.fold_feature] != fold].drop(self.fold_feature, axis=1)
            x_valid = self.x.loc[self.x[self.fold_feature] == fold].drop(self.fold_feature, axis=1)
            y_valid = self.y.loc[self.y[self.fold_feature] == fold].drop(self.fold_feature, axis=1)
            self.models[fold].fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=0, early_stopping_rounds=100)

            full_preds[y_valid.index] = self.models[fold].predict_proba(x_valid)[:, 1]

        return full_preds

    def objective(self, trial):

        param = {
            "objective": trial.suggest_categorical("objective", ["Logloss"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        preds = self.train(param)

        return accuracy_score(self.y.drop(self.fold_feature, axis=1), np.rint(preds))

    def fit(self, num_trials=100):
        if self.params is None:
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=num_trials)
            self.params = study.best_trial.params

        preds = self.train(self.params)
        print(f'CatBoostScore: {accuracy_score(self.y.drop(self.fold_feature, axis=1), np.rint(preds))}')

        df = pd.DataFrame(preds)
        df.columns = ['catboost']

        return df

    def predict(self, x):
        preds = pd.DataFrame()
        preds.index = range(len(x))
        preds['catboost'] = 0.0
        for i in range(self.num_folds):
            preds['catboost'] = preds['catboost'] + pd.Series(self.models[i].predict_proba(x)[:, 1])

        preds['catboost'] = preds['catboost'] / float(self.num_folds)

        return preds

