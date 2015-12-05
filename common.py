import random
import zipfile
import numpy as np
import pandas as pd


def import_training_data(file_name):
    print("Importing " + file_name)
    zip = zipfile.ZipFile("input/"+file_name+'.zip')

    data = pd.read_csv(zip.open(file_name))

    keeps = ["Dates", "Category", "DayOfWeek", "PdDistrict", "X", "Y", "Address"]
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
    return data.drop(["Id"], axis=1)


def discretize(data, col):
    print("\tDiscretizing " + col)
    val_list = list(enumerate(np.unique(data[col])))
    val_dict = {name: i for i, name in val_list}
    return data[col].map(lambda x: val_dict[x]).astype(int)


def cross_validation(no_folds, train_X, train_Y):
    data = pd.concat([train_X, train_Y], axis=1)
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


def write_result(output, method):
    print("Write results of " + method)
    output.to_csv("output/" + method + "_submit.csv", index_label="Id")
    print("Done.")

