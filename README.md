
# Azure MLOps Project: Diabetes Prediction Model

## Project Overview
This project demonstrates the full MLOps lifecycle for training, registering, and deploying a machine learning model using Azure Machine Learning (AzureML). The model predicts diabetes progression based on provided data using the Ridge regression algorithm.

## Key Features
- **Model Training**: Ridge regression model is trained using the Diabetes dataset from `scikit-learn`.
- **Model Registration**: The trained model is registered in Azure's model registry for versioning and later deployment.
- **Model Deployment**: The registered model is deployed for real-time inference using Azure's endpoints.
- **Azure Environment Setup**: Configured a custom environment to handle dependencies like `scikit-learn`, `joblib`, and `numpy`.

## Project Structure
- **Model Training and Registration**:
  - The `Ridge` regression model is trained on the diabetes dataset.
  - The model is saved as `model_ridge.pkl` and registered in the AzureML workspace.

- **Inference Script**:
  - A scoring script (`score2.py`) is created to handle predictions. It loads the registered model and processes incoming data to return predictions in JSON format.

- **Environment Setup**:
  - A custom Azure environment is created with necessary dependencies (`numpy`, `joblib`, `scikit-learn`).

## Prerequisites
- Azure subscription and an AzureML workspace.
- Python 3.x environment.
- AzureML SDK installed (`pip install azureml-sdk`).

## Steps to Run the Project
1. **Authentication**:
   - Ensure that you are authenticated with Azure by setting up the `config.json` file for your AzureML workspace.
   
   ```python
   from azureml.core import Workspace
   ws = Workspace.from_config()
   ```

2. **Model Training**:
   - Load the diabetes dataset, train the Ridge regression model, and save it.
   
   ```python
   from sklearn.datasets import load_diabetes
   from sklearn.linear_model import Ridge
   import joblib

   x, y = load_diabetes(return_X_y=True)
   model = Ridge().fit(x, y)
   joblib.dump(model, "model_ridge.pkl")
   ```

3. **Model Registration**:
   - Register the trained model in AzureML.

   ```python
   from azureml.core.model import Model
   model = Model.register(model_path="model_ridge.pkl", model_name="model_ridge", workspace=ws)
   ```

4. **Deployment**:
   - Create a scoring script (`score2.py`) for real-time inference.
   - Set up the environment and deploy the model.

   ```python
   from azureml.core import Environment, InferenceConfig
   env = Environment("deploytocloudenv")
   env.python.conda_dependencies.add_pip_package("joblib")
   env.python.conda_dependencies.add_pip_package("numpy==1.23")
   env.python.conda_dependencies.add_pip_package("scikit-learn==1.3.1")

   inference_config = InferenceConfig(entry_script="score2.py", environment=env)
   ```

## Model Inference
- After deployment, the model can handle requests by accepting input data and returning predictions in JSON format.

## Dependencies
- `azureml-core`
- `scikit-learn`
- `numpy`
- `joblib`

## License
This project is licensed under the MIT License.
