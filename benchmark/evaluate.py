import pandas as pd
import numpy as np

import os

from surprise import Reader, Dataset
from surprise import accuracy, dump
np.random.seed(42)

CUR_FILE_FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.join(CUR_FILE_FOLDER, '..')

READER = Reader(rating_scale=(1,5))
TRAIN_DATA = Dataset.load_from_df(
    pd.read_csv(os.path.join('..', 'data', 'interim', 'train_data.csv'), header=0, sep='\t'),
    READER).build_full_trainset()

def make_prediction(uid, model, num=5):
    '''
    Function gives the arbitrary number of the films recommended to the user
    '''
    predictions = []
    for ii in TRAIN_DATA.all_items():
        ii = TRAIN_DATA.to_raw_iid(ii)
        predictions.append(model.predict(uid, ii, verbose = False))
    return [x.iid for x in sorted(predictions, key=lambda x: x.est, reverse=True)[:num]]

def load_eval_data():
    '''
    Loading the evaluation dataset for testing the model
    '''
    data = pd.read_csv('data/test_data.csv', header=0, sep='\t')
    
    sup_data = Dataset.load_from_df(data, READER)
    return sup_data.construct_testset(sup_data.raw_ratings)


if __name__ == "__main__":
    print('Loading the model...')
    _, model = dump.load(os.path.join(ROOT_FOLDER, 'models', 'surprise_svd'))

    print('Loading the test data...')
    eval_data = load_eval_data()

    print("\n###################TESTING###################\n")

    predictions = model.test(eval_data, verbose=0)
    print("Accuracy scores:")
    accuracy.rmse(predictions)
    accuracy.mse(predictions)
    accuracy.mae(predictions)

    user_id = 300 # You can put here different user to see test it for different users
    print()
    print(f"Predicted sample of top 5 films for user {user_id}")
    print(make_prediction(user_id, model))
