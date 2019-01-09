"""
Defines the class which will be used as a model to make predictions on the 
Kickstarter dataset to determine whether a Kickstarter will be successfully
funded or not.
"""
import json
import pickle
from datetime import datetime

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

COLUMNS_FILENAME = "./.columns.npy" 


def encoding_labels(labels):
    """
    Will take the set of labels (y) for a classification problem and check if 
    they have been encoded into integers. If so, then leave them be, if not they
    will be encoded as needed and returned.

    Labels : Pandas Series of labels
    Output : Encoded Pandas Series
    """
    state_encodings = {v:k for k, v in enumerate(labels.unique())}
    return labels.apply(lambda x: state_encodings[x])


def drop_empty_cols(df):
    """
    Drops columns from a dataframe using the heuristic of more than 20% of the
    rows in the column being unusued.
    """
    df_clean = df.copy()
    try:
        drop_cols = [k for k, v in dict(df_clean.isna().sum()).items() 
                     if v > int(df_clean.shape[0] * 0.2)]
    except AttributeError:
        drop_cols = [k for k, v in dict(df_clean.isnull().sum()).items() 
                     if v > int(df_clean.shape[0] * 0.2)]
    df_clean.drop(drop_cols, axis=1, inplace=True) 
    return df_clean.dropna()


def unixtime_to_datetime(ts):
    """
    Converts UNIX timestamp to Python datetime object.
    """
    return datetime.fromtimestamp(float(ts))


class KickstarterModel:

    def __init__(self):
        nltk.download('stopwords')
        languages = ['english', 'german', 'french', 'spanish', 'italian']
        stopwords_list = stopwords.words(languages)

        def nlp_model_gen(ngram_range):
            model = Pipeline([
                ('vect', CountVectorizer(stop_words=stopwords_list, 
                                         analyzer="word",
                                         ngram_range=(1, 4),
                                         token_pattern="[a-zA-Z]{2,}", min_df=1,
                                         strip_accents="ascii")), 
                ('tfidf', TfidfTransformer()),
                ('clf', LinearSVC(loss="hinge", max_iter=100, tol=1e-10))
            ])
            return model

        # nlp_models = [(str(i), nlp_model_gen((1, i))) for i in range(2, 5)]
        # self.__nlp_model = VotingClassifier(estimators=nlp_models)
        self.__nlp_model = nlp_model_gen((1, 4))  # VotingClassifier(estimators=nlp_models)

        self.__model = GradientBoostingClassifier(max_depth=3, n_estimators=100,
                                                  learning_rate=0.1)

        # self.__stacked_model = LogisticRegression(solver='lbfgs')

    def __ensemble_data_split(self, df):
        nlp_data = df.blurb

    def __clean_data(self, df):
        keep_cols = ['country', 'category', 'static_usd_rate', 'goal', 'blurb',
                     'deadline', 'launched_at', 'location', 'state']
        df_main = df.copy().loc[:, keep_cols]

        # Static USD Rate 
        convert_to_usd = lambda x: x.static_usd_rate * x.goal
        cols = ["static_usd_rate", "goal"]
        df_main.loc[:, "goal_adjusted"] = df_main.loc[:, cols].apply(
            convert_to_usd, axis=1)
        df_main.goal_adjusted.head()

        # Unix timestamp to Python datetime
        for col in ["deadline", "launched_at"]:
            df_main.loc[:, col] = df_main.loc[:, col].apply(unixtime_to_datetime)

        # Duration of Fund Raiser
        duration = df_main.deadline - df_main.launched_at
        df_main.loc[:, "duration"] = duration.apply(lambda x: x.days)

        # Get actual category from the JSON object in category column.
        def get_category(x):
            category = json.loads(x)["slug"]    
            return pd.Series(category.split("/"))
                    
        categories = df_main.category.apply(get_category)
        categories.rename(columns={0: "topic", 1: "sub_category"}, inplace=True)
        df_main = df_main.merge(categories, left_index=True, right_index=True,
                                how='outer')

        # Final Column Select
        final_cols = ["country", "duration", "goal_adjusted",
                      "topic", "sub_category", "state", "blurb"]
        df_final = df_main.loc[:, final_cols]
        return df_final

    def __one_hot_encode(self, df):
        # One-hot encoding
        df_tmp = df.copy()
        one_hot_cols = ["country", "topic", "sub_category"]
        return pd.get_dummies(df_tmp, columns=one_hot_cols)

    def preprocess_training_data(self, df):
        X, y = self.__preprocess_data(df)

        # Finals Columns
        # print("FINAL COLS:", len(X.columns.values))
        np.save(COLUMNS_FILENAME, X.columns.values)

        return X, y

    def __preprocess_data(self, df):
        # Split the dataframe into 2 separate dataframes
        # Pass the individual dataframes into 2 separate parasing functions
        df_ = drop_empty_cols(df)

        # Binary Outcomes
        df_.state = encoding_labels(df_.state)

        # Data transformations to clean up data
        df_clean = self.__clean_data(df_)
        expt_cols = ["country", "duration", "goal_adjusted",
                     "topic", "sub_category", "state", "blurb"]

        # One-hot encoding
        df_clean_oh_encoded = self.__one_hot_encode(df_clean)

        # Features, Labels
        y = df_.state
        X = df_clean_oh_encoded.drop("state", axis=1)

        return X, y  

    def fit(self, X, y):
        X_fit = X.copy()
        X_nlp = X_fit.blurb
        X_other = X_fit.drop("blurb", axis=1)
        y_enc = encoding_labels(y)

        # for _, model in self.__nlp_models:
        # model.fit(X_nlp, y_enc)
        
        self.__nlp_model.fit(X_nlp, y_enc)
        self.__model.fit(X_other, y_enc)

    def preprocess_scoring_data(self, df):
        X, y = self.__preprocess_data(df)
        all_cols = np.load(COLUMNS_FILENAME)

        # Finding all the columns which haven't been produced as a result of 
        # one hot encoding the test data, since the test data will not be as 
        # large as the training data.
        missed_cols = [c for c in all_cols if c not in X.columns]
        print("missed_cols:", len(missed_cols))

        # Adding the missed columns to the dataframe.
        for c in missed_cols:
            X[c] = pd.Series(0, index=X.index)
        
        # Put columns in the same order as training.
        X_final = X[all_cols]

        # Putting the columns in the same order as training.
        return X_final, y

    def preprocess_unseen_data(self, df):
        return self.preprocess_scoring_data(df)[0]

    def predict(self, X):
        X_pred = X.copy()
        X_nlp = X_pred.blurb
        print(X_nlp.head())
        X_other = X_pred.drop("blurb", axis=1)

        y_pred_nlp = self.__nlp_model.predict(X_nlp)
        # y_pred = self.__model.predict(X_other)
        return y_pred_nlp  # , y_pred)

        # X_stacked = np.stack([y_nlp_pred, y_pred], axis=1)

        # return self.__stacked_model.predict(X_stacked)

    def score(self, X, y):
        print("score")
        # y_pred1, y_pred2 = self.predict(X)
        y_pred = self.predict(X)
        print(list(y_pred[0:10]))
        # print(list(y_pred2[0:10]))
        print(list(y[0:10]))
        print()
        # return (accuracy_score(y_pred, y), accuracy_score(y_pred2, y))
        return accuracy_score(y_pred, y)  # , accuracy_score(y_pred2, y))
