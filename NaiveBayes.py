from common import *
import pandas as pd
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
    model.fit(train_X, train_Y)

    print("Predicting the result")
    test_Y = model.predict_proba(test_X)
    test_Y = pd.DataFrame(test_Y, index=test_X.index, columns=model.classes_)
    print(test_Y[:10])
    write_result(test_Y, 'NaiveBayes')


def preprocess(data):
    data["PdDistrict"] = discretize(data, "PdDistrict")
    data["DayOfWeek"] = discretize(data, "DayOfWeek")

    print("\tSeparating Hour")
    data["Hour"] = data["Dates"].map(lambda x: pd.to_datetime(x).hour)
    print("\tSeparating Month")
    data["Month"] = data["Dates"].map(lambda x: pd.to_datetime(x).month)
    print("\tSeparating Year")
    data["Year"] = data["Dates"].map(lambda x: pd.to_datetime(x).year)

    print("\tMake Coordinates positive")
    data["X"] = data["X"].map(lambda x: -x)

    return data.drop("Dates", axis=1)


if __name__ == "__main__":
    main()
