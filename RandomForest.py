from common import *
from sklearn.ensemble import RandomForestClassifier


def main():
    train_X, train_Y = import_training_data('train.csv')
    test_X = import_testing_data('test.csv')

    print("Preprocessing Training Data...")
    train_X = preprocess(train_X)
    print("Preprocessing Testng Data...")
    test_X = preprocess(test_X)

    print("Fitting model")
    model = RandomForestClassifier(n_estimators=20)
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
        results.append(pd.DataFrame(model.predict_proba(test_X), columns=model.classes_))

        i += 1
        print("\t" + str(i) + "th fold\t" + str(accuracy) + "%")

    print("Generating combined model")
    concat_data = pd.concat(results)
    test_Y = concat_data.groupby(concat_data.index)
    test_Y = test_Y.head()

    print("Predicting the result")
    print(test_Y[:10])
    print(test_Y.count, test_Y.columns.values)
    write_result(test_Y, "RandomForest")


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


if __name__ == "__main__":
    main()
