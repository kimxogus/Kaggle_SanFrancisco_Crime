from common import *
from sklearn.ensemble import RandomForestClassifier


def main():
    train_X, train_Y = import_training_data('train.csv')
    test_X = import_testing_data('test.csv')

    print("Preprocessing Training Data...")
    train_X = preprocess(train_X)
    print("Preprocessing Testng Data...")
    test_X = preprocess(test_X)
    print("Generate grid")
    train_X, test_X = generate_grid(train_X, test_X, no_grid=50)

    print("Fitting model")
    model = RandomForestClassifier(n_estimators=20)
    columns = []
    results = []
    i = 0
    for train, test in cross_validation(10, train_X, train_Y):
        tr_Y = train["Category"]
        tr_X = train.drop("Category", axis=1)
        ts_Y = test["Category"]
        ts_X = test.drop("Category", axis=1)

        model.fit(tr_X, tr_Y)
        output = model.predict(ts_X)
        accuracy = float(sum(output == ts_Y))/len(ts_Y)*100
        results.append(pd.DataFrame(model.predict_proba(test_X), index=test_X.index, columns=model.classes_))
        columns = model.classes_

        i += 1
        print("\t" + str(i) + "th fold\t" + str(accuracy) + "%")

    print("Generating combined model")
    test_Y = pd.DataFrame(index=test_X.index, columns=columns)
    for col in columns:
        print("Calculating " + col + "'s mean")
        lst = []
        for data in results:
            lst.append(data[col])
        data = pd.concat(lst, axis=1)
        test_Y[col] = data.mean(axis=1)

    print(test_Y[:5])
    write_result(test_Y, "RandomForest")


def preprocess(data):
    data["PdDistrict"] = discretize(data, "PdDistrict")
    data["DayOfWeek"] = discretize(data, "DayOfWeek")

    print("\tExtracting Dates")
    dates = data["Dates"].map(lambda x: pd.to_datetime(x))
    print("\tSeparating Hour")
    data["Hour"] = dates.map(lambda x: x.hour)
    print("\tSeparating Month")
    data["Month"] = dates.map(lambda x: x.month)
    print("\tSeparating Year")
    data["Year"] = dates.map(lambda x: x.year)

    return data.drop(["Dates", "Address"], axis=1)


def generate_grid(train, test, no_grid=100):
    for dim in ["X", "Y"]:
        print("\tBinning " + dim)
        tr = train[dim].map(lambda x: abs(x))
        ts = test[dim].map(lambda x: abs(x))

        top = np.max([tr.max(), ts.max()])
        bottom = np.min([tr.min(), ts.min()])

        bins = np.linspace(bottom, top, no_grid)

        train[dim] = pd.cut(tr, bins, labels=range(1, no_grid), include_lowest=True).astype(int)
        test[dim] = pd.cut(ts, bins, labels=range(1, no_grid), include_lowest=True).astype(int)

    for data in [train, test]:
        data["XandY"] = data[["X", "Y"]].apply(lambda x: x["X"]*x["Y"], axis=1)
        data["XorY"] = data[["X", "Y"]].apply(lambda x: x["X"]+x["Y"], axis=1)
        data = data.drop(["X", "Y"], axis=1)

    return train, test


if __name__ == "__main__":
    main()
