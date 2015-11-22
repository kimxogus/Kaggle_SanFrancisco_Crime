import zipfile
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def main():
    train_X, train_Y = import_training_data('train.csv')
    test_X = import_testing_data('test.csv')

    print("Preprocessing...")
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)

    print(train_X.columns.values, test_X.columns.values)
    print("Fitting model")
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(train_X, train_Y)

    print("Predicting the result")
    test_Y = forest.predict_proba(test_X)
    test_Y = pd.DataFrame(test_Y, index=test_X.index, columns=forest.classes_)
    print(test_Y)


def import_training_data(file_name):
    print("Importing " + file_name)
    zip = zipfile.ZipFile("input/"+file_name+'.zip')

    data = pd.read_csv(zip.open(file_name))

    keeps = ["Id", "Dates", "Category", "DayOfWeek", "PdDistrict", "X", "Y"]
    drops = []
    for col in data.columns.values:
        if col not in keeps:
            drops.append(col)

    data = data.drop(drops,  axis=1)

    return data.drop("Category", axis=1), data["Category"]


def import_testing_data(file_name):
    print("Importing " + file_name)
    zip_file = zipfile.ZipFile("input/"+file_name+'.zip')

    data = pd.read_csv(zip_file.open(file_name))
    data = pd.DataFrame(data, index=data["Id"])
    return data.drop(["Id", "Address"], axis=1)


def preprocess(data):
    data["PdDistrict"] = discretize(data, "PdDistrict")
    data["DayOfWeek"] = discretize(data, "DayOfWeek")

    print("\tSeparating Hour")
    data["Hour"] = data["Dates"].map(lambda x: pd.to_datetime(x).hour)
    print("\tSeparating Month")
    data["Month"] = data["Dates"].map(lambda x: pd.to_datetime(x).month)
    print("\tSeparating Year")
    data["Year"] = data["Dates"].map(lambda x: pd.to_datetime(x).year)

    return data.drop("Dates", axis=1)


def discretize(data, col):
    print("\tDiscretizing " + col)
    val_list = list(enumerate(np.unique(data[col])))
    val_dict = {name: i for i, name in val_list}
    return data[col].map(lambda x: val_dict[x]).astype(int)


def cross_validation(data, no_folds=10):
    rows = list(data.index)
    random.shuffle(rows)
    n = len(data)
    folded_len = int(n / no_folds)
    begin = 0
    for i in range(no_folds):
        if i == folded_len - 1:
            end = n
        else:
            end = begin + folded_len
        test = data.ix[rows[begin:end]]
        train = data.ix[rows[:begin] + rows[end:]]
        yield [test, train]
        begin = end


if __name__ == "__main__":
    main()
