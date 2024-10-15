# Customer Churn Prediction using AWS SageMaker

This project demonstrates how to build, train, and deploy a machine learning model to predict customer churn using Amazon SageMaker. The model is trained using the XGBoost algorithm and leverages AWS services like S3 and SageMaker for training and inference.

## Project Structure

- **data/**: Contains the dataset used for training (gym_churn_no_header.csv).
- **scripts/**: Python scripts for data preprocessing, training the model, and deployment.
  - `train.py`: Script to train the XGBoost model on SageMaker.
- **churn-analysis.ipynb**: Jupyter notebook that provides an end-to-end walkthrough of the churn analysis process, including data preprocessing, training, and predictions.

## Steps to Run the Project

### 1. Set Up the Environment

Before running the project, ensure you have the following prerequisites:

- AWS credentials with appropriate permissions for SageMaker and S3.
- A SageMaker execution role ARN.

### 2. Clone the Repository

```bash
git clone https://github.com/JUnelus/Customer-Churn-Prediction-using-AWS-SageMaker.git
cd Customer-Churn-Prediction-using-AWS-SageMaker
```

### 3. Train the Model
To train the model on SageMaker, run the `train.py` script. This script uploads the dataset to S3, configures an XGBoost training job on SageMaker, and deploys the trained model to an endpoint.
```bash
python scripts/train.py
```

### 4. Perform Predictions
Once the model is deployed, you can use the SageMaker endpoint to make predictions on new data. The test data should match the number of features used during training.

### 5. Clean Up
To avoid incurring additional costs, remember to delete the SageMaker endpoint once you're done with predictions.
```bash
xgb_predictor.delete_endpoint()
```

## Requirements
- Python 3.11
- SageMaker Python SDK
- Pandas
- Boto3
- Python-dotenv
```bash
- pip install -r requirements.txt
```
![img.png](img.png)