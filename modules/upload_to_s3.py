def upload(sagemaker_session, bucket, prefix, file_path='data/data.csv'):
    '''
    Data will be stored in the same path inside the bucket.
    Returns data location.
    '''

    # upload csv to S3 bucket
    
    raw_data = sagemaker_session.upload_data(
        path="{}".format(file_path),
        bucket=bucket,
        key_prefix="{}".format(prefix))
        
    print('Data has been stored in the following bucket:', bucket)
    
    return raw_data



