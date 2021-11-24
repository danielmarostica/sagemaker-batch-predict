#!/usr/bin/env python  
import argparse
import joblib
import os

from io import StringIO

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-estimators', type=int, default=10)
    parser.add_argument('--min-samples-leaf', type=int, default=3)

    # variáveis de ambiente do sagemaker, que podem ser substituídas por parâmetros ao executar localmente
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--train-file', type=str, default='train.csv')
    parser.add_argument('--test-file', type=str, default='test.csv')

    args, _ = parser.parse_known_args()

    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    X_train = train_df.iloc[:, 1:]
    X_test = test_df.iloc[:, 1:]
    y_train = train_df.iloc[:, 0]
    y_test = test_df.iloc[:, 0]

    model = RandomForestClassifier(
        n_estimators=args.n_estimators, min_samples_leaf=args.min_samples_leaf, n_jobs=-1)

    model.fit(X_train, y_train)

    path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, path)
    print('Model saved at ' + path)

def model_fn(model_dir):
    '''
    O nome desta função é especial e será reconhecido pelo SageMaker para inferências.
    '''
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model

# def input_fn(input_data, content_type):
#     """
#     Para inferência, aceitar somente csv.
#     """
#     if content_type == 'text/csv':
#         df = pd.read_csv(StringIO(input_data), header=None)
#         return df
#     else:
#         raise ValueError("{} not supported by this script.".format(content_type))