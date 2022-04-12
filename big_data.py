
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
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def load_data():
    """
    Load data from txt file to pandas dataframe
    """
    train_df = pd.read_csv("train_data.csv", sep="\t")
    test_df = pd.read_csv("test_data.csv", sep="\t")
    return train_df, test_df


def train_predict_feature_importance(models, train_df, test_df, target="Class", name="Unknown"):
    """
    Train and predict for each model and calculate feature importance
    """

    for model in models:
        model.fit(train_df.drop([target], axis=1), train_df[target])
        predictions = model.predict(test_df.drop([target], axis=1))
        accuracy = accuracy_score(test_df[target], predictions)
        print(
            f"DataSet: {name} Model: {type(model).__name__} Accuracy: {accuracy}")
        if type(model) == RandomForestClassifier:
            plt.title("Feature importances" + name)
            plt.bar(train_df.drop([target], axis=1).columns,
                    model.feature_importances_, color="r", align="center")
            plt.xticks(rotation=90)
            plt.savefig("feature_importances_" + name + ".png")


if __name__ == "__main__":
    """
    Compare at least 3 different classification methods in terms of predictive performance. Carefully
    consider how to deal with the imbalance in the data ­which performance metrics should you use?
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

    models = [RandomForestClassifier(),
              KNeighborsClassifier(),
              AdaBoostClassifier()]

    train_predict_feature_importance(
        models, train_df, test_df, name="unbalanced")

    rus = RandomUnderSampler(random_state=42)

    train_df_sm, target_df_sm = rus.fit_resample(
        train_df.drop([target], axis=1), train_df[target])

    train_df_sm = pd.concat((train_df_sm, target_df_sm), axis=1)

    train_predict_feature_importance(
        models, train_df_sm, test_df, name="Balanced_upsample")

    sm = SMOTE(random_state=42)

    train_df_sm, target_df_sm = sm.fit_resample(
        train_df.drop([target], axis=1), train_df[target])

    train_df_sm = pd.concat((train_df_sm, target_df_sm), axis=1)

    train_predict_feature_importance(
        models, train_df_sm, test_df, name="Balanced_downsample")
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
