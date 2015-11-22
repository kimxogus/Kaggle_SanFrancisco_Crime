import zipfile
import numpy as np
import pandas as pd

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


def discretize(data, col):
    print("\tDiscretizing " + col)
    val_list = list(enumerate(np.unique(data[col])))
    val_dict = {name: i for i, name in val_list}
    return data[col].map(lambda x: val_dict[x]).astype(int)


def write_result(output, method):
    output.to_csv("output/" + method + "_submit.csv", index_label="Id")

