import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from train import one_hot, drop_data, mk_submission
from sklearn.model_selection import KFold

def train(data, true_str):
    x_train = data.drop([true_str], axis=1)
    y_train = data['Survived']

    kf = KFold(n_splits=5)
    score_list = []
    models = []

    for fold, (train_index, valid_index) in enumerate(kf.split(x_train, y_train)):
        train_x = x_train.iloc[train_index]
        valid_x = x_train.iloc[valid_index]
        train_y = y_train[train_index]
        valid_y = y_train[valid_index]

        print(f'flod{fold + 1} start')

        model = lgb.LGBMClassifier(objective='binary')
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

    models = train(train_data, 'Survived')
    pred = test(models, test_data)

    mk_submission(pred, '../results/kfold_test_predict_result.csv')


if __name__ == '__main__':
    main()