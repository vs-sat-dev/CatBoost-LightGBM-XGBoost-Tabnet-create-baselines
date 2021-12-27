import pandas as pd
from sklearn.model_selection import StratifiedKFold


def generate_folds(data, target_feature, fold_feature, num_folds, seed=0):
    kfold = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)
    for e, (trn_ind, val_ind) in enumerate(kfold.split(data.drop(target_feature, axis=1),
                                                       data[target_feature])):
        data.loc[val_ind, fold_feature] = e

    return data


def factorize_cat_features(data, factorize_map=None):
    for column in data.columns:
        if data[column].dtype == 'object':
            if factorize_map:
                data.loc[:, column] = data[column].replace(factorize_map[column])
            else:
                data.loc[:, column] = data.loc[:, column].factorize()[0]
    return data


def get_data(path, target_feature=None, fold_feature=None, num_folds=5):
    df = pd.read_csv(path)

    df = df.drop('Loan_ID', axis=1)
    df.loc[:, 'Dependents'] = df.loc[:, 'Dependents'].replace('3+', '3').fillna(0).astype('int')

    if target_feature and fold_feature:
        df[target_feature] = df[target_feature].replace({'N': 0, 'Y': 1})
        df = generate_folds(df, target_feature, fold_feature, num_folds)

    df = factorize_cat_features(df)

    return df
