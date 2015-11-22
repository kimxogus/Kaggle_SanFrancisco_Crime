import zipfile
import random
import numpy as np
import pandas as pd


def main():
    train_X, train_Y = import_training_data('train.csv')
    #test_X = import_testing_data('test.csv')

    print("Preprocessing...")
    print(preprocess(train_X[:5]))


def import_training_data(file_name):
    print("Importing " + file_name)
    zip = zipfile.ZipFile("input/"+file_name+'.zip')

    data = pd.read_csv(zip.open(file_name))

    keeps = ["Id", "Dates", "Category", "DayOfWeek", "PdDistrict", "Address", "X", "Y"]
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

    return data


def preprocess(data):
    district = discritize(data, "PdDistrict")
    dayOfWeek = discritize(data, "DayOfWeek")

    hours = data["Dates"].map(lambda x: pd.to_datetime(x).hour)
    months = data["Dates"].map(lambda x: pd.to_datetime(x).month)
    years = data["Dates"].map(lambda x: pd.to_datetime(x).year)

    data.drop(["Dates", "PdDistrict", "DayOfWeek"], axis=1)

    return pd.concat([data, hours, months, years, district, dayOfWeek], axis=1)


def discritize(data, col):
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
