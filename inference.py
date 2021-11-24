import sys
import os

import boto3, sagemaker

from sagemaker.sklearn.model import SKLearnModel
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ProcessingInput, ProcessingOutput

sys.path.append('modules')

# start session
sagemaker_session = sagemaker.Session()

# get role with sagemaker, s3, redshift permissions
iam = boto3.client('iam')
role = iam.get_role(RoleName='datascience-sagemaker-s3-redshift')['Role']['Arn']

# bucket folder name (prefix)
bucket = sagemaker_session.default_bucket()

# preprocessing data
print('Starting preprocessing job')

# get data for inference
data = os.path.join('s3://', bucket, 'titanic_example', 'data.csv')

sklearn_processor = SKLearnProcessor(framework_version='0.23-1',
                                    role=role,
                                    instance_type='ml.t3.medium',
                                    instance_count=1)

sklearn_processor.run(code='modules/preprocessing.py',
                    inputs=[ProcessingInput(
                            source=data,
                            destination='/opt/ml/processing/input')],
                    outputs=[ProcessingOutput(output_name='processed_data',
                                                source='/opt/ml/processing/data')],
                    arguments=['--inference', 'true'])

preprocessing_job_description = sklearn_processor.jobs[-1].describe() 
processed_data = os.path.join('s3://', bucket, preprocessing_job_description['ProcessingJobName'], 'output', 'processed_data', 'data.csv')     

# processed_data = os.path.join('s3://', bucket, 'sagemaker-scikit-learn-2021-11-23-22-02-37-569', 'output', 'processed_data', 'data.csv')   

# load training job name
with open ("training_job_name.txt", "r") as job:
    job_name = job.read().splitlines()[0]

# load model for inference
model_artifact = os.path.join('s3://', bucket, job_name, 'output', 'model.tar.gz')

model = SKLearnModel(model_data=model_artifact,
                     role=role,
                     framework_version='0.23-1',
                     entry_point='modules/model.py')

transformer = model.transformer(
    instance_count=1, instance_type="ml.m4.xlarge", assemble_with="Line", accept="text/csv")                          

prediction = transformer.transform(data=processed_data, content_type='text/csv')
print(processed_data)
print(prediction)
