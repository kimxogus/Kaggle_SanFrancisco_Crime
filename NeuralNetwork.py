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
    model = MLPClassifier(verbose=True, activation='logistic', batch_size=300,
                          algorithm='adam', early_stopping=False)

    model.fit(train_X, train_Y)
    test_Y = pd.DataFrame(model.predict_proba(test_X), index=test_X.index, columns=model.classes_)

    print(test_Y[:5])
    write_result(test_Y, 'NeuralNetwork_CLP_adam')


def preprocess(data):
    print("\tMaking dummy data of PdDistrict")
    district = pd.get_dummies(data["PdDistrict"], prefix='district_')
    print("\tMaking dummy data of DayOfWeek")
    day = pd.get_dummies(data["DayOfWeek"], prefix='day_')

    print("\tExtracting Dates")
    dates = data["Dates"].map(lambda x: pd.to_datetime(x))
    print("\t\tSeparating Hour")
    hour = pd.get_dummies(dates.map(lambda x: x.hour), prefix='hour_')
    print("\t\tSeparating Month")
    month = pd.get_dummies(dates.map(lambda x: x.month), prefix="month_")
    print("\t\tSeparating Year")
    year = pd.get_dummies(dates.map(lambda x: x.year), prefix="year_")

    print("\tDrop columns")
    data = data.drop(["Dates", "PdDistrict", "DayOfWeek"], axis=1)

    print("\tConcatenate Original and dummy data")
    data = pd.concat([data, district, day, hour, month, year], axis=1)

    return data


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

    x = pd.get_dummies(train["X"], prefix="X_")
    y = pd.get_dummies(train["Y"], prefix="Y_")
    train = train.drop(["X", "Y"], axis=1)
    train = pd.concat([train, x, y], axis=1)

    x = pd.get_dummies(test["X"], prefix="X_")
    y = pd.get_dummies(test["Y"], prefix="Y_")
    test = test.drop(["X", "Y"], axis=1)
    test = pd.concat([test, x, y], axis=1)

    return train, test


if __name__ == "__main__":
    main()
