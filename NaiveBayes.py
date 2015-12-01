from common import *
from sklearn.naive_bayes import MultinomialNB


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
    model = MultinomialNB(alpha=0.01)
    columns = []
    results = []
    i = 0
    for fold in cross_validation(10, train_X, train_Y):
        tr_Y = fold[1]["Category"]
        tr_X = fold[1].drop("Category", axis=1)
        ts_Y = fold[0]["Category"]
        ts_X = fold[0].drop("Category", axis=1)

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
    write_result(test_Y, 'NaiveBayes')


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

    for data in [train, test]:
        data["XandY"] = data[["X", "Y"]].apply(lambda x: x["X"]*x["Y"], axis=1)
        data["XorY"] = data[["X", "Y"]].apply(lambda x: x["X"]+x["Y"], axis=1)
        data = data.drop(["X", "Y"], axis=1)

    return train, test


if __name__ == "__main__":
    main()
