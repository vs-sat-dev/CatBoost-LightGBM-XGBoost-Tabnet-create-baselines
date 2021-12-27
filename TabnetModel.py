from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.metrics import accuracy_score
import optuna
import numpy as np
import pandas as pd


class TabnetModel:
    def __init__(self, x, y, num_folds, fold_feature, params=None, models=None):
        self.num_folds = num_folds
        self.fold_feature = fold_feature
        self.params = params
        self.models = models
        self.x = self.data_preprocessing(x)
        self.y = y

    def data_preprocessing(self, x):
        new_x = x.copy()
        for column in x.columns:
            new_x[column] = x[column].fillna(x[column].mode()[0])
            if column != self.fold_feature:
                max_val = new_x[column].max()
                if max_val > 1:
                    new_x[column] = new_x[column] / max_val
        return new_x

    def train(self, params):
        self.models = [TabNetClassifier(**params) for _ in range(self.num_folds)]

        full_preds = np.zeros(len(self.y))

        for fold in range(self.num_folds):
            X_train = self.x.loc[self.x[self.fold_feature] != fold].drop(self.fold_feature, axis=1).values
            y_train = self.y.loc[self.y[self.fold_feature] != fold].drop(self.fold_feature, axis=1).values.flatten()
            X_valid = self.x.loc[self.x[self.fold_feature] == fold].drop(self.fold_feature, axis=1).values
            y_valid = self.y.loc[self.y[self.fold_feature] == fold].drop(self.fold_feature, axis=1).values.flatten()

            self.models[fold].fit(X_train=X_train, y_train=y_train, eval_set=[(X_valid, y_valid)],
                                  patience=30, max_epochs=100, eval_metric=['accuracy'])

            full_preds[self.y.loc[self.y[self.fold_feature] == fold].index] = \
                self.models[fold].predict_proba(X_valid)[:, 1]

        return full_preds

    def objective(self, trial):

        mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        n_d = trial.suggest_int("n_d", 56, 64, step=4)
        n_steps = trial.suggest_int("n_steps", 1, 3, step=1)
        gamma = trial.suggest_float("gamma", 1., 1.4, step=0.2)
        n_shared = trial.suggest_int("n_shared", 1, 3)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)

        param = dict(n_d=n_d, n_a=n_d, n_steps=n_steps, gamma=gamma,
                     lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type=mask_type, n_shared=n_shared,
                     scheduler_params=dict(mode="min", min_lr=1e-5, factor=0.5,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau, verbose=0)

        preds = self.train(param)

        return accuracy_score(self.y.drop(self.fold_feature, axis=1), np.rint(preds))

    def fit(self, num_trials=100):
        if self.params is None:
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=num_trials)
            self.params = study.best_trial.params
            print(f'params: {self.params}')

        preds = self.train(self.params)
        print(f'TabnetScore: {accuracy_score(self.y.drop(self.fold_feature, axis=1), np.rint(preds))}')

        df = pd.DataFrame(preds)
        df.columns = ['tabnet']

        return df

    def predict(self, x):
        new_x = self.data_preprocessing(x)
        preds = pd.DataFrame()
        preds.index = range(len(x))
        preds['tabnet'] = 0.0
        for i in range(self.num_folds):
            preds['tabnet'] = preds['tabnet'] + pd.Series(self.models[i].predict_proba(new_x.values)[:, 1])

        preds['tabnet'] = preds['tabnet'] / float(self.num_folds)

        return preds
