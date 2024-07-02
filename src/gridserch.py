
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

FOLD_NUM = 5


def search_params(x_train, y_train):
    model = lgb.LGBMClassifier(objective='binary')

    print('\nparams: ', model.get_params())

    params = {
        'max_depth' : [2, 3, 4, 5],
        'reg_alpha' : [0, 1, 10, 50, 100],
        'reg_lambda': [0, 1, 10, 50, 100],
    }

    grid_serch = GridSearchCV(
        model,
        param_grid=params,
        cv=FOLD_NUM
    )

    grid_serch.fit(x_train, y_train)

    print('BEST score: ',grid_serch.best_score_)
    print('BEST params: ',grid_serch.best_params_)

    return grid_serch.best_params_


def train(data, true_str):
    x_train = data.drop([true_str], axis=1)
    y_train = data['Survived']

    params = search_params(x_train, y_train)

    kf = KFold(n_splits=FOLD_NUM)
    score_list = []
    models = []
    for fold, (train_index, valid_index) in enumerate(kf.split(x_train, y_train)):
        train_x = x_train.iloc[train_index]
        valid_x = x_train.iloc[valid_index]
        train_y = y_train[train_index]
        valid_y = y_train[valid_index]

        print(f'flod{fold + 1} start')

        model = lgb.LGBMClassifier(objective='binary', max_depth=params['max_depth'], reg_alpha=params['reg_alpha'], reg_lambda=params['reg_lambda'])
        model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)])

        pred = model.predict(valid_x, num_iteration=model.best_iteration_)
        score_list.append(round(accuracy_score(valid_y, pred)*100, 2))
        models.append(model)

        print(f'fold{fold + 1} end\n')
    
    print(score_list, 'Ave Score: ', np.mean(score_list), '%')

    return models

def test(models, test_data):
    
    preds = np.zeros((len(test_data), 5))

    for fold, model in enumerate(models):
        pred_ = model.predict(test_data, num_iteration=model.best_iteration_)
        preds[:, fold] = pred_
    
    pred = (np.mean(preds, axis=1) > 0.5).astype(int)

    return pred

def drop_data(data, *strs):
    strs_list = list(strs)
    data.drop(strs_list, axis=1, inplace=True)
    return data

def one_hot(data, str1, str2):
    data = pd.get_dummies(data, columns=[str1, str2])
    return data

def mk_submission(pred, file_path):
    submission = pd.read_csv('../data/titanic_csv_data/gender_submission.csv')
    submission['Survived'] = pred
    submission.to_csv(file_path, index=False)

def main():
    print('start')
    train_data = pd.read_csv('../data/titanic_csv_data/train.csv')
    test_data = pd.read_csv('../data/titanic_csv_data/test.csv')
    print('\nrow train data: \n', train_data.head())
    print(' row test data: \n', test_data.head())

    train_data = one_hot(train_data, 'Sex', 'Embarked')
    test_data = one_hot(test_data, 'Sex', 'Embarked')

    train_data = drop_data(train_data, 'PassengerId', 'Name', 'Cabin', 'Ticket')
    test_data = drop_data(test_data, 'PassengerId', 'Name', 'Cabin', 'Ticket')
    print('--------------------------------------------')
    print('\nfixed train data:\n ',train_data.head())
    print('\nfixed test data:\n ',test_data.head())

    model = train(train_data, 'Survived')
    pred = test(model, test_data)

    mk_submission(pred, '../results/grid_predict_result.csv')

if __name__ == '__main__':
    main()