import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from dataset import get_data
from CatBoostModel import CatBoostModel
from LGBModel import LGBModel
from XGBModel import XGBModel
from TabnetModel import TabnetModel


def create_submission(submission_name, target_feature, preds):
    test_submission = pd.DataFrame()
    test_submission['Loan_ID'] = pd.read_csv('data/test.csv')['Loan_ID']
    test_submission[target_feature] = np.rint(preds)
    test_submission[target_feature] = test_submission[target_feature].replace({0: 'N', 1: 'Y'})
    test_submission.to_csv(submission_name, index=False)


if __name__ == '__main__':
    target_feature = 'Loan_Status'
    fold_feature = 'fold'
    num_folds = 5
    num_trials = 250
    df_train = get_data('data/train.csv', target_feature=target_feature, fold_feature=fold_feature)
    df_test = get_data('data/test.csv')

    results_train = df_train[[fold_feature, target_feature]].copy()

    catboost_model = CatBoostModel(df_train.drop(target_feature, axis=1),
                                   df_train[[target_feature, fold_feature]],
                                   num_folds=num_folds, fold_feature=fold_feature)
    results_train = pd.concat([results_train, catboost_model.fit(num_trials=num_trials)], axis=1)
    preds = catboost_model.predict(df_test)
    create_submission('catboost_submission.csv', target_feature, preds)
    results_test = preds

    lgb_model = LGBModel(df_train.drop(target_feature, axis=1),
                         df_train[[target_feature, fold_feature]],
                         num_folds=num_folds, fold_feature=fold_feature)
    results_train = pd.concat([results_train, lgb_model.fit(num_trials=num_trials)], axis=1)
    preds = lgb_model.predict(df_test)
    create_submission('lgb_submission.csv', target_feature, preds)
    results_test = pd.concat([results_test, preds], axis=1)

    xgb_model = XGBModel(df_train.drop(target_feature, axis=1),
                         df_train[[target_feature, fold_feature]],
                         num_folds=num_folds, fold_feature=fold_feature)
    results_train = pd.concat([results_train, xgb_model.fit(num_trials=num_trials)], axis=1)
    preds = xgb_model.predict(df_test)
    create_submission('xgb_submission.csv', target_feature, preds)
    results_test = pd.concat([results_test, preds], axis=1)

    tabnet_model = TabnetModel(df_train.drop(target_feature, axis=1),
                               df_train[[target_feature, fold_feature]],
                               num_folds=num_folds, fold_feature=fold_feature)
    results_train = pd.concat([results_train, tabnet_model.fit(num_trials=num_trials)], axis=1)
    preds = tabnet_model.predict(df_test)
    create_submission('tabnet_submission.csv', target_feature, preds)
    results_test = pd.concat([results_test, preds], axis=1)

    models = [LogisticRegression() for _ in range(num_folds)]
    test_preds = np.zeros(len(results_test))
    for fold in range(num_folds):
        X_train = results_train.loc[results_train[fold_feature] != fold].drop([target_feature, fold_feature], axis=1)
        X_test = results_train.loc[results_train[fold_feature] == fold].drop([target_feature, fold_feature], axis=1)
        y_train = results_train.loc[results_train[fold_feature] != fold][target_feature]
        y_test = results_train.loc[results_train[fold_feature] == fold][target_feature]

        models[fold].fit(X_train, y_train)
        preds = models[fold].predict_proba(X_test)[:, 1]

        print(f'AssembleScore: {accuracy_score(y_test, np.rint(preds))}')

        test_preds = test_preds + models[fold].predict_proba(results_test)[:, 1]

    test_preds = test_preds / float(num_folds)
    print(f'test_preds: {test_preds}')

    create_submission('submission.csv', target_feature, test_preds)
