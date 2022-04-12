
"""
Question 1

Compare and train models for classification as well as to identify important predictors.
Note, the data is imbalanced.

"""

import pandas as pd

# import classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_data():
    """
    Load data from txt file to pandas dataframe
    """
    train_df = pd.read_csv(".BigData/train_data.csv", sep="\t")
    test_df = pd.read_csv(".BigData/test_data.csv", sep="\t")
    return train_df, test_df


def train_predict_feature_importance(models, train_df, test_df):
    """
    Train and predict for each model and calculate feature importance
    """
    target = "Class"

    for model in models:
        model.fit(train_df.drop(target), train_df[target])
        predictions = model.predict(test_df.drop(target))
        accuracy = accuracy_score(test_df[target], predictions)
        print(f"Accuracy {type(model)}: {accuracy}")


if __name__ == "__main__":
    """
    Compare at least 3 different classification methods in terms of predictive performance. Carefully
    consider how to deal with the imbalance in the data ­ which performance metrics should you use?
    how should models be trained? should the data be re­balanced and if so how? At the conclusion of
    this task you should be able to say something about relative model performance on the test data.
    """

    train_df, test_df = load_data()

    target = "Class"

    # count the number of each class
    print("Number of each class in training set:")
    print(train_df[target].value_counts())

    print("Number of each class in test set:")
    print(test_df[target].value_counts())

    rf_classifier = RandomForestClassifier()
    knn_classifier = KNeighborsClassifier()
    ada_boost_classifier = AdaBoostClassifier()

    rf_classifier.fit(train_df.drop([target], axis=1), train_df[target])
    knn_classifier.fit(train_df.drop([target], axis=1), train_df[target])
    ada_boost_classifier.fit(train_df.drop([target], axis=1), train_df[target])

    rf_pred = rf_classifier.predict(test_df.drop([target], axis=1))
    knn_pred = knn_classifier.predict(test_df.drop([target], axis=1))
    ada_boost_pred = ada_boost_classifier.predict(
        test_df.drop([target], axis=1))

    # metrics
    print("RF Accuracy:", accuracy_score(test_df[target], rf_pred))
    print("KNN Accuracy:", accuracy_score(test_df[target], knn_pred))
    print("Ada Boost Accuracy:", accuracy_score(
        test_df[target], ada_boost_pred))

    from matplotlib import pyplot as plt
    plt.title("Feature importances Unbalanced")
    plt.bar(train_df.drop(target, axis=1).columns,
            rf_classifier.feature_importances_, color="r", align="center")
    plt.xticks(rotation=90)
    plt.savefig(".BigData/feature_importances_unbalanced.png")

    # balance the data
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    rus = RandomUnderSampler(random_state=42)

    train_df_sm, target_df_sm = rus.fit_resample(
        train_df.drop([target], axis=1), train_df[target])

    # train the models
    rf_classifier.fit(train_df_sm, target_df_sm)
    knn_classifier.fit(train_df_sm, target_df_sm)
    ada_boost_classifier.fit(train_df_sm, target_df_sm)

    rf_pred = rf_classifier.predict(test_df.drop([target], axis=1))
    knn_pred = knn_classifier.predict(test_df.drop([target], axis=1))
    ada_boost_pred = ada_boost_classifier.predict(
        test_df.drop([target], axis=1))

    # metrics
    print("RF Accuracy Balanced DOWN:",
          accuracy_score(test_df[target], rf_pred))
    print("KNN Accuracy Balanced DOWN:",
          accuracy_score(test_df[target], knn_pred))
    print("Ada Boost Accuracy Balanced DOWN:", accuracy_score(
        test_df[target], ada_boost_pred))

    from matplotlib import pyplot as plt
    plt.title("Feature importances Down Sampel")
    plt.bar(train_df.drop(target, axis=1).columns,
            rf_classifier.feature_importances_, color="r", align="center")
    plt.xticks(rotation=90)
    plt.savefig(".BigData/feature_importances_DOWN.png")

    sm = SMOTE(random_state=42)

    train_df_sm, target_df_sm = sm.fit_resample(
        train_df.drop([target], axis=1), train_df[target])

    # train the models
    rf_classifier.fit(train_df_sm, target_df_sm)
    knn_classifier.fit(train_df_sm, target_df_sm)
    ada_boost_classifier.fit(train_df_sm, target_df_sm)

    rf_pred = rf_classifier.predict(test_df.drop([target], axis=1))
    knn_pred = knn_classifier.predict(test_df.drop([target], axis=1))
    ada_boost_pred = ada_boost_classifier.predict(
        test_df.drop([target], axis=1))

    from matplotlib import pyplot as plt
    plt.title("Feature importances Down Sampel")
    plt.bar(train_df.drop(target, axis=1).columns,
            rf_classifier.feature_importances_, color="r", align="center")
    plt.xticks(rotation=90)
    plt.savefig(".BigData/feature_importances_UP.png")

    # metrics
    print("RF Accuracy Balanced UP:", accuracy_score(test_df[target], rf_pred))
    print("KNN Accuracy Balanced UP:",
          accuracy_score(test_df[target], knn_pred))
    print("Ada Boost Accuracy Balanced UP:", accuracy_score(
        test_df[target], ada_boost_pred))

    # Feature importance
    print(
        f"RF Feature Importance: {rf_classifier.feature_importances_ > 0.05}")

    """
    Your next task is to identify important features for prediction. Motivate which methods you choose
    for feature selection. Be careful to evalute how confident you are about the selection of features.
    Does the class imbalance impact the selection? Is this affected by re­balancing of the classes?
    """

    """
    Create a 3 class problem from your 2 class problem by selecting a subset of features and splitting
    the minority class into two in these features. You can do this by adding a adjusting the mean value
    to the feature for example or by simple splitting the feature at some threshold value.
    """

    # create a 3 class problem
    train_df_3class = train_df.copy()
    # train_df_3class['Class'] = train_df_3class['Class'].apply(
    # split the minority class into two in these features
