import pandas as pd 
import numpy as np 

import json
import logging
import os
import sys

#CONFIGURE EXPERIMENT
import mlflow
import dagshub
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "yogibaba7"
repo_name = "ci_supporter_loan_approval"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# Initialize client
client = MlflowClient()
model_name = "mymodel"

# archieve existing production model
def archieve_production(client ,model_name:str)->None:
    # Check if there's already a version with alias 'production'
    try:
        production_version_info = client.get_model_version_by_alias(model_name, "production")
        production_version = production_version_info.version

        # Remove 'production' alias from current prod version
        client.delete_registered_model_alias(model_name, "production")

        # Assign alias 'archive' to the old production version
        client.set_registered_model_alias(model_name, "archive", production_version)

    except Exception as e:
        print("No existing production model found, skipping archive step.")

# now staging is done lets move it to production
def promote_to_production(client,model_name:str)->None:
    try:
        # Get current version with alias 'staging'
        staging_version_info = client.get_model_version_by_alias(model_name, "staging")
        staging_version = staging_version_info.version

        # Promote staging model to production
        client.set_registered_model_alias(model_name, "production", staging_version)

        # Optional: remove staging alias now that it's promoted
        #client.delete_registered_model_alias(model_name, "staging")
    except Exception as e:
        print("No existing staging model found, skipping promote production step.")


if __name__=="__main__":
    archieve_production(client,model_name)
    promote_to_production(client,model_name)





