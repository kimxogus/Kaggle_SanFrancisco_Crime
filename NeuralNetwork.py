from common import *
from sklearn.neural_network import MLPClassifier


def main():
    train_X, train_Y = import_training_data('train.csv')
    test_X = import_testing_data('test.csv')

    print("Preprocessing Training Data...")
    train_X = preprocess(train_X)
    print("Preprocessing Testng Data...")
    test_X = preprocess(test_X)
    print("Generate grid")
    train_X, test_X = generate_grid(train_X, test_X, no_grid=50)
    print(train_X[:5])
    print(test_X[:5])

    print("Fitting model")
    model = MLPClassifier(verbose=True, activation='logistic', shuffle=True, tol=5*1e-4,
                        algorithm='adam', random_state=0)

    model.fit(train_X, train_Y[train_X.index])

    print(np.sum(model.predict(train_X) == train_Y[train_X.index])*10000/len(train_X.index)/100)

    test_Y = pd.DataFrame(model.predict_proba(test_X), index=test_X.index, columns=model.classes_)

    print(test_Y[:5])
    write_result(test_Y, 'RBM_MLP')


def preprocess(data):
    print("\tBinarizing PdDistrict")
    district = pd.get_dummies(data["PdDistrict"], prefix='district')
    print("\tBinarizing DayOfWeek")
    day = pd.get_dummies(data["DayOfWeek"], prefix='day')

    print("\tExtracting Dates")
    dates = data["Dates"].map(lambda x: pd.to_datetime(x))
    print("\t\tSeparating Hour")
    hour = pd.get_dummies(dates.map(lambda x: x.hour), prefix='hour')
    print("\t\tSeparating Month")
    month = pd.get_dummies(dates.map(lambda x: x.month), prefix="month")
    print("\t\tSeparating Year")
    year = pd.get_dummies(dates.map(lambda x: x.year), prefix="year")

    print("\tGenerating corner data")
    corner = data["Address"].apply(lambda x: 1 if '/' in x else 0)

    print("\tDrop columns")
    data = data.drop(["Dates", "PdDistrict", "DayOfWeek", "Address"], axis=1)

    print("\tArranging data")
    data = pd.concat([data, district, corner, day, hour, month, year], axis=1)

    return data


def generate_grid(train, test, no_grid=100):
    print("\tDrop invalid locations in training data")
    train = train[train["X"] < -122][train["X"] > -123]
    train = train[train["Y"] < 38][train["Y"] > 37]

    for dim in ["X", "Y"]:
        print("\tBinning " + dim)
        tr = train[dim]
        ts = test[dim]
        if dim == "X":
            ts = ts[ts < -122][ts > -123]
        elif dim == "Y":
            ts = ts[ts < 38][ts > 37]

        top = np.max([tr.max(), ts.max()])
        bottom = np.min([tr.min(), ts.min()])
        bins = np.linspace(bottom, top, no_grid)

        train[dim] = pd.cut(tr, bins, labels=range(1, no_grid), include_lowest=True).astype(int)
        test[dim] = pd.cut(ts, bins, labels=range(1, no_grid), include_lowest=True).astype(int)

    x = pd.get_dummies(train["X"], prefix="X")
    y = pd.get_dummies(train["Y"], prefix="Y")
    train = train.drop(["X", "Y"], axis=1)
    train = pd.concat([train, x, y], axis=1)

    x = pd.get_dummies(test["X"], prefix="X")
    y = pd.get_dummies(test["Y"], prefix="Y")
    test = test.drop(["X", "Y"], axis=1)
    test = pd.concat([test, x, y], axis=1)

    return train, test


if __name__ == "__main__":
    main()
