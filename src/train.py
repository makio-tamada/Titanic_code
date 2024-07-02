
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

def train(data, true_str):
    x_train = data.drop([true_str], axis=1)
    y_train = data['Survived']

    train_x, valid_x, train_y, valid_y = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    model = lgb.LGBMClassifier(objective='binary')
    model.fit(train_x, train_y, eval_set= [(valid_x, valid_y)])

    #predict for valid_x
    pred = model.predict(valid_x, num_iteration=model.best_iteration_)
    print('Nomal lightGBM Score', round(accuracy_score(valid_y,pred)*100, 2), '%')

    return model

def test(model, test_data):
    pred = model.predict(test_data, num_iteration=model.best_iteration_)
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

    mk_submission(pred, '../results/test_predict_result.csv')

if __name__ == '__main__':
    main()