import xgboost as xgb
from sklearn.metrics import accuracy_score
import optuna
import numpy as np
import pandas as pd


class XGBModel:
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

            dtrain = xgb.DMatrix(x_train, label=y_train)
            dvalid = xgb.DMatrix(x_valid, label=y_valid)

            self.models[fold] = xgb.train(params, dtrain)

            full_preds[y_valid.index] = self.models[fold].predict(dvalid)

        return full_preds

    def objective(self, trial):

        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            # use exact for small dataset.
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        preds = self.train(param)

        return accuracy_score(self.y.drop(self.fold_feature, axis=1), np.rint(preds))

    def fit(self, num_trials=100):
        if self.params is None:
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=num_trials)
            self.params = study.best_trial.params

        preds = self.train(self.params)
        print(f'XGBScore: {accuracy_score(self.y.drop(self.fold_feature, axis=1), np.rint(preds))}')

        df = pd.DataFrame(preds)
        df.columns = ['xgb']

        return df

    def predict(self, x):
        preds = pd.DataFrame()
        preds.index = range(len(x))
        preds['xgb'] = 0.0
        dpred = xgb.DMatrix(x)
        for i in range(self.num_folds):
            preds['xgb'] = preds['xgb'] + pd.Series(self.models[i].predict(dpred))

        preds['xgb'] = preds['xgb'] / float(self.num_folds)

        return preds

