import argparse
import os
import pickle
import urllib.request

import pandas as pd
import numpy as np

from model import KickstarterModel as Model
from model import drop_empty_cols 

DATASET_URL = "https://s3-eu-west-1.amazonaws.com/kate-datasets/kickstarter/train.zip"
DATA_DIR = "data"
DATA_FILENAME = "train.zip"
PICKLE_NAME = "model.pickle"
TRAIN_FRAC = 0.8


def setup_data():
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    req = urllib.request.urlopen(DATASET_URL)
    data = req.read()

    with open(os.path.join(DATA_DIR, DATA_FILENAME), "wb") as f:
        f.write(data)


def train_model():
    df = pd.read_csv(os.sep.join([DATA_DIR, DATA_FILENAME]))
    train_size = int(len(df)*TRAIN_FRAC)
    df = df.iloc[:train_size, :]

    my_model = Model()
    X_train, y_train = my_model.preprocess_training_data(df)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    my_model.fit(X_train, y_train)

    # Save to pickle
    with open(PICKLE_NAME, 'wb') as f:
        pickle.dump(my_model, f)


def test_model():
    df = pd.read_csv(os.sep.join([DATA_DIR, DATA_FILENAME]))
    train_size = int(len(df)*TRAIN_FRAC)
    df = df.iloc[train_size:, :]

    # Load pickle
    with open(PICKLE_NAME, 'rb') as f:
        my_model = pickle.load(f)

    X_test = my_model.preprocess_unseen_data(df)
    preds = my_model.predict(X_test)
    print("### Your predictions ###")
    print(preds)


def score_model():
    df = pd.read_csv(os.sep.join([DATA_DIR, DATA_FILENAME]))
    train_size = int(len(df)*TRAIN_FRAC)
    train = df.iloc[:train_size, :]
    test = df.iloc[train_size:, :]

    # Load pickle
    with open(PICKLE_NAME, 'rb') as f:
        my_model = pickle.load(f)

    X_train, y_train = my_model.preprocess_scoring_data(train)
    X_test, y_test = my_model.preprocess_scoring_data(test)

    train_score = my_model.score(X_train, y_train)
    test_score = my_model.score(X_test, y_test)
    print("### Model Score ###")
    print(f"Train Accuracy: {train_score}")
    print(f"Test Accuracy: {test_score}")


def main():
    parser = argparse.ArgumentParser(
        description="A command line-tool to manage the project.")
    parser.add_argument(
        'stage',
        metavar='stage',
        type=str,
        choices=['setup', 'train', 'test', 'score'],
        help="Stage to run.")

    stage = parser.parse_args().stage

    if stage == "setup":
        print("Downloading dataset...")
        setup_data()
    elif stage == "train":
        print("Training model...")
        train_model()
    elif stage == "test":
        print("Testing model...")
        test_model()
    elif stage == "score":
        print("Evaluating model...")
        score_model()


if __name__ == "__main__":
    main()
