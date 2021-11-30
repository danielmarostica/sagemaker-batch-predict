#!/usr/bin/env python

import os
import sys
sys.path.append('modules')

from upload_to_s3 import upload
import boto3, sagemaker

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ProcessingInput, ProcessingOutput

# get role with sagemaker, s3, permissions
iam = boto3.client('iam')
role = iam.get_role(RoleName='datascience-sagemaker-s3-redshift')['Role']['Arn']

# start session (you must have your credentials file at /home/your.name/.aws/ correctly set with CLI keys and tokens)
sagemaker_session = sagemaker.Session()

# bucket folder name (prefix)
bucket = sagemaker_session.default_bucket() # creates a bucket based on your region and account ID
prefix = "titanic_example" # folder name

# upload csv to s3
s3_data_uri = upload(sagemaker_session, bucket, prefix, file_path='data/data.csv')

# preprocessing data
print('Starting preprocessing job') # you can check its state at AWS's web interface under SageMaker > Processing Jobs
sklearn_processor = SKLearnProcessor(framework_version='0.23-1',
                                    role=role,
                                    instance_type='ml.t3.medium',
                                    instance_count=1,
                                    base_job_name='sm-preprocessing')

sklearn_processor.run(code='modules/preprocessing.py',
                    inputs=[ProcessingInput(
                            source=s3_data_uri,
                            destination='/opt/ml/processing/input')],
                    outputs=[ProcessingOutput(output_name='train_data',
                                                source='/opt/ml/processing/train'),
                            ProcessingOutput(output_name='test_data',
                                                source='/opt/ml/processing/test')],
                    arguments=['--train-test-split-ratio', '0.2'])

preprocessing_job_description = sklearn_processor.jobs[-1].describe() # save the name of the processing job

# training
print('Starting training job')
sklearn = SKLearn(
    entry_point='modules/model.py',
    framework_version='0.23-1',
    instance_type='ml.m5.large',
    role=role,
    sagemaker_session=sagemaker_session,
    hyperparameters={"max_leaf_nodes": 30},
    base_job_name='sm-training')

train_file = os.path.join('s3://', bucket, preprocessing_job_description['ProcessingJobName'], 'output', 'train_data', 'train.csv')
test_file = os.path.join('s3://', bucket, preprocessing_job_description['ProcessingJobName'], 'output', 'test_data', 'test.csv')

sklearn.fit({'train': train_file, 'test': test_file})

training_job_name = sklearn._current_job_name # save the name of the training job             

with open("training_job_name.txt", "w") as text_file:
    text_file.write(f'{training_job_name}')



