import boto3
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize a session with boto3 to interact with AWS
s3 = boto3.client('s3')

# Define S3 bucket and key paths
bucket = os.getenv('AWS_S3_BUCKET_NAME')  # Replace with your actual S3 bucket name
prefix = 'sagemaker/gym-churn'  # Folder name where data will be stored
file_path = r'C:\Users\big_j\PycharmProjects\Customer-Churn-Prediction-using-AWS-SageMaker\data\gym_churn_us.csv'  # Path to your file

# Upload the churn data to S3
data_key = f'{prefix}/gym_churn_us.csv'
s3.upload_file(file_path, bucket, data_key)
print(f"File uploaded to s3://{bucket}/{data_key}")

# Set up SageMaker session and role
sagemaker_session = sagemaker.Session()
role = os.getenv('AWS_ROLE_ARN')  # Replace with your actual IAM role ARN

# Define the S3 location of the input data
input_data = f's3://{bucket}/{data_key}'

# Set up the container for the built-in XGBoost algorithm
container = get_image_uri(sagemaker_session.boto_region_name, 'xgboost')

# Initialize the XGBoost estimator
xgb = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type='ml.m5.large',  # Use for general purpose and lower cost
    output_path=f's3://{bucket}/{prefix}/output',
    sagemaker_session=sagemaker_session
)

# Set hyperparameters for the XGBoost model
xgb.set_hyperparameters(
    objective="binary:logistic",
    num_round=100,
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    silent=0
)

# Define input/output channels
train_input = sagemaker.inputs.TrainingInput(
    s3_data=input_data,
    content_type='csv'
)

# Train the model
xgb.fit({'train': train_input})

# Deploy the model to an endpoint
xgb_predictor = xgb.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Modify the test data to be in CSV format with 13 values
test_data = '0.2,0.1,0.4,0.5,0.6,0.7,0.8,0.3,0.9,0.5,0.4,0.2,0.1'

# Make the prediction, ensuring the content type is set to 'text/csv'
prediction = xgb_predictor.predict(test_data, initial_args={'ContentType': 'text/csv'})

print('Predicted class:', prediction)

# Delete the endpoint after inference
xgb_predictor.delete_endpoint()
