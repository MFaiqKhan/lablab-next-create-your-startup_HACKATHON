import subprocess
import sagemaker
from sagemaker import get_execution_role
import boto3
import time
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



# Define Triton container and create SageMaker model
sm_client = boto3.client("sagemaker", region_name=region)
region = boto3.Session().region_name

account_id_map = {
    "us-east-1": "785573368785",
    "us-east-2": "007439368137",
    "us-west-1": "710691900526",
    "us-west-2": "301217895009",
}

triton_image_uri = f"{account_id_map[region]}.dkr.ecr.{region}.amazonaws.com/sagemaker-tritonserver:21.08-py3"

sm_model_name = "triton-inceptionv3-pt-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

container = {
    "Image": triton_image_uri,
    "ModelDataUrl": model_uri,
    "Environment": {"SAGEMAKER_TRITON_DEFAULT_MODEL_NAME": "inceptionv3"},
}

create_model_response = sm_client.create_model(
    ModelName=sm_model_name, ExecutionRoleArn=role, PrimaryContainer=container
)

print("Model Arn: " + create_model_response["ModelArn"])

# Create Endpoint Configuration
endpoint_config_name = "triton-inceptionv3-pt-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

create_endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "InstanceType": "ml.c5.2xlarge",
            "InitialVariantWeight": 1,
            "InitialInstanceCount": 1,
            "ModelName": sm_model_name,
            "VariantName": "AllTraffic",
        }
    ],
)

print("Endpoint Config Arn: " + create_endpoint_config_response["EndpointConfigArn"])

# Create SageMaker Endpoint
endpoint_name = "triton-inceptionv3-pt-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

create_endpoint_response = sm_client.create_endpoint(
    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
)

print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])

# Wait for endpoint creation
resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
status = resp["EndpointStatus"]
print("Status: " + status)

while status == "Creating":
    time.sleep(60)
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    print("Status: " + status)

print("Arn: " + resp["EndpointArn"])
print("Status: " + status)

print("SageMaker Endpoint is deployed and ready for inference.")
