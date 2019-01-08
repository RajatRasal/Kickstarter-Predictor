"""
Defines the class which will be used as a model to make predictions on the 
Kickstarter dataset to determine whether a Kickstarter will be successfully
funded or not.
"""
import json
import numpy as np
import nltk
import pandas as pd
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


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


def unixtimestamp_to_datetime(ts):
    """
    Converts UNIX timestamp to Python datetime object.
    """
    return datetime.fromtimestamp(float(ts))


class KickstarterModel:

    def __init__(self):
        nltk.download('stopwords')
        languages = ['english', 'german', 'french', 'spanish', 'italian']
        stopwords_list = stopwords.words(languages)

        self.__nlp_model = Pipeline([
            ('vect', CountVectorizer(stop_words=stopwords_list, analyzer="word", 
                                     token_pattern="[a-zA-Z]{2,}", min_df=1,
                                     strip_accents="ascii", ngram_range=(1, 4))),
            ('tfidf', TfidfTransformer()),
            ('clf-svm', LinearSVC(loss="hinge", max_iter=100, random_state=42, 
                                  tol=1e-10))
        ])
        self.__non_text_model = None

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
            df_main.loc[:, col] = df_main.loc[:, col].apply(unixtimestamp_to_datetime)

        # Duration of Fund Raiser
        df_main.loc[:, "duration"] = (df_main.deadline - df_main.launched_at).apply(lambda x: x.days)

        # Get actual category from the JSON in categ
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

        return df_clean_oh_encoded.drop("state", axis=1), df_.state


    def fit(self, X, y):
        X_fit = X.copy()
        X_nlp = X_fit.blurb
        X_other = X_fit.drop("blurb", axis=1)

        y = encoding_labels(y)

        self.__nlp_model.fit(X_nlp, y)

    def preprocess_unseen_data(self, df):
        return self.preprocess_training_data(df)[0]

    def predict(self, X):
        X_pred = X.copy()
        X_nlp = X_pred.blurb
        X_other = X_pred.drop("blurb", axis=1)
        return self.__nlp_model.predict(X_nlp)

    def score(self, X, y):
        return accuracy_score(self.predict(X), y)
