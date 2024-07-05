import subprocess
import sagemaker
import boto3
from dotenv import load_dotenv
import os

load_dotenv()

region_aws = os.getenv("REGION_AWS")
role_aws = os.getenv("ROLE_AWS")

region = region_aws
sagemaker_session = sagemaker.Session(boto3.Session(region_name=region))
role = role_aws


subprocess.run(['tar', '-czf', 'model.tar.gz', '-C' , 'models', 'inceptionv3'], check=True)

# Upload the tarball to a specific S3 bucket
bucket_name = "hackathons3bucket"
model_uri = sagemaker_session.upload_data(path="model.tar.gz", bucket=bucket_name, key_prefix="triton-model")
print(f'Model uploaded to: {model_uri}')