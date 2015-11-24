from common import *
from sklearn.naive_bayes import MultinomialNB


def main():
    train_X, train_Y = import_training_data('train.csv')
    test_X = import_testing_data('test.csv')

    print("Preprocessing Training Data...")
    train_X = preprocess(train_X)
    print("Preprocessing Testng Data...")
    test_X = preprocess(test_X)

    print("Fitting model")
    model = MultinomialNB(alpha=0.00001)
    columns = []
    accuracies = []
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
        accuracies.append(accuracy)
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

    print(test_Y)
    write_result(test_Y, 'NaiveBayes')


def preprocess(data):
    print("\tMaking dummy data of PdDistrict")
    PdDistrict = pd.get_dummies(data["PdDistrict"])
    print("\tMaking dummy data of DayOfWeek")
    DayOfWeek = pd.get_dummies(data["DayOfWeek"])

    print("\tExtracting Dates")
    Dates = data["Dates"].map(lambda x: pd.to_datetime(x))
    print("\t\tSeparating Hour")
    Hour = pd.get_dummies(Dates.map(lambda x: x.hour))
    print("\t\tSeparating Month")
    Month = pd.get_dummies(Dates.map(lambda x: x.month))
    print("\t\tSeparating Year")
    Year = pd.get_dummies(Dates.map(lambda x: x.year))

    print("\tMake Coordinates positive")
    data["X"] = data["X"].map(lambda x: abs(x))

    print("\tDrop columns")
    data = data.drop(["Dates", "PdDistrict", "DayOfWeek"], axis=1)

    print("\tConcatenate Original and dummy data")
    data = pd.concat([data, PdDistrict, DayOfWeek, Hour, Month, Year], axis=1)

    return data


if __name__ == "__main__":
    main()
