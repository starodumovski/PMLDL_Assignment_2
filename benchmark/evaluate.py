import pandas as pd
import numpy as np

import os
import sys

from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy, dump
np.random.seed(42)

CUR_FILE_FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.join(CUR_FILE_FOLDER, '..')

# sys.path.insert(0, os.path.join(ROOT_FOLDER, 'models'))

def make_prediction(uid, model, num=5):
    predictions = []
    for ii in train_data.all_items():
        ii = train_data.to_raw_iid(ii)
        predictions.append(model.predict(uid, ii, verbose = False))
    return [x.iid for x in sorted(predictions, key=lambda x: x.est, reverse=True)[:num]]


if __name__ == "__main__":
    _, model = dump.load(os.path.join(ROOT_FOLDER, 'models', 'surprise_svd'))