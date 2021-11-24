import argparse
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    parser.add_argument('--input-folder', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--inference', type=str, default='false')
    args, _ = parser.parse_known_args()

    print('Received arguments {}'.format(args))

    input_data_path = os.path.join(args.input_folder, 'data.csv')

    print('Reading input data from {}'.format(input_data_path))
    df = pd.read_csv(input_data_path)

    if args.inference == 'false': # for training
        split_ratio = args.train_test_split_ratio
        print('Splitting data into train and test sets with ratio {}'.format(split_ratio))

        X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), df['Survived'], test_size=split_ratio, random_state=0)

        preprocess = make_column_transformer(
            (StandardScaler(), ['Age', 'Fare', 'Parch', 'SibSp', 'Family_Size']),
            (OneHotEncoder(sparse=False), ['Embarked', 'Pclass', 'Sex']))

        print('Running preprocessing and feature engineering transformations')
        X_train = preprocess.fit_transform(X_train)
        X_test = preprocess.transform(X_test)

        df_train = pd.DataFrame(data=np.column_stack((y_train, X_train))) # first line must be the target
        df_test = pd.DataFrame(data=np.column_stack((y_test, X_test)))

        df_train.rename(columns={df_train.columns[0]: 'Survived'}, inplace=True)
        df_test.rename(columns={df_test.columns[0]: 'Survived'}, inplace=True)

        train_output_path = os.path.join('/opt/ml/processing/train', 'train.csv')
        test_output_path = os.path.join('/opt/ml/processing/test', 'test.csv')

        df_train.to_csv(train_output_path, header=False, index=False)
        df_test.to_csv(test_output_path, header=False, index=False)

    else:
        preprocess = make_column_transformer(
            (StandardScaler(), ['Age', 'Fare', 'Parch', 'SibSp', 'Family_Size']),
            (OneHotEncoder(sparse=False), ['Embarked', 'Pclass', 'Sex']))
        
        processed_data = preprocess.fit_transform(pd.concat([df['Survived'], df.drop('Survived', axis=1)], axis=1)) # make target the first column
        df_processed_data = pd.DataFrame(data=processed_data)
        df_processed_data.rename(columns={df_processed_data.columns[0]: 'Survived'}, inplace=True)
        
        data_output_path = os.path.join('/opt/ml/processing/data', 'data.csv')
        df_processed_data.to_csv(data_output_path, header=False, index=False)