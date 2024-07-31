"""
----------------------------------------------------------------------------------------
exe_3_logit_MLflow.py

Demonstrates how to override the predict method in a custom model using MLflow's PythonModel API.

: zach.wolpe@medibio.com.au
: 30-07-24
-----
"""

from joblib import dump
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

import os
import shutil

def empty_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

class ModelWrapper(PythonModel):
    def __init__(self):
        self.model = None

    def load_context(self, context):
        from joblib import load

        self.model = load(context.artifacts["model_path"])

    def predict(self, context, model_input, params=None):
        params = params or {"predict_method": "predict"}
        predict_method = params.get("predict_method")

        if predict_method == "predict":
            return self.model.predict(model_input)
        elif predict_method == "predict_proba":
            return self.model.predict_proba(model_input)
        elif predict_method == "predict_log_proba":
            return self.model.predict_log_proba(model_input)
        else:
            raise ValueError(f"The prediction method '{predict_method}' is not supported.")


if __name__ == '__main__':
    mlflow.set_tracking_uri("http://localhost:8080")

    # load data
    iris = load_iris()
    x = iris.data[:, 2:]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9001)

    # train model
    model = LogisticRegression(random_state=0, max_iter=5_000, solver="newton-cg").fit(x_train, y_train)

    # evaluate model/predict
    print('Prediction: \n', model.predict(x_test)[:5])
    print('Predictive Probability: \n', model.predict_proba(x_test)[:5])
    print('Score: ', model.score(x_test, y_test))

    # use model outside of python environment
    mlflow.set_experiment("Overriding Predict Tutorial")
    sklearn_path = "./tmp/sklearn_model"

    # clear directory if required
    empty_dir(sklearn_path)

    # save model 1: sklearn without predict_proba
    with mlflow.start_run() as run:
        mlflow.sklearn.save_model(
            sk_model=model,
            path=sklearn_path,
            input_example=x_train[:2],
        )

    # load and predict (only .predict is instantiated)
    loaded_logreg_model = mlflow.pyfunc.load_model(sklearn_path)
    y_pred = loaded_logreg_model.predict(x_test)
    print('Prediction: \n', y_pred[:5])

    # force exception
    # try:
    #     loaded_logreg_model.predict_proba(x_test)
    # except ValueError as e:
    #     print(f"Exception: {e}")

    # serialize the model
    model_directory = "./tmp/sklearn_model.joblib"
    dump(model, model_directory)

    # load a custom model artifact
    # Define the required artifacts associated with the saved custom pyfunc
    artifacts = {"model_path": model_directory}

    # Define the signature associated with the model
    # signature = infer_signature(x_train, params={"predict_method": "predict_proba"})

    # print("Signature: ")
    # print(signature)

    pyfunc_path = "./tmp/dynamic_regressor"
    empty_dir(pyfunc_path)

    # Save the custom model to the specified path
    with mlflow.start_run() as run:
        mlflow.pyfunc.save_model(
            path=pyfunc_path,
            python_model=ModelWrapper(),
            input_example=x_train,
            # signature=signature,
            artifacts=artifacts,
            pip_requirements=["joblib", "sklearn"],
        )

    # Load the custom model
    loaded_dynamic = mlflow.pyfunc.load_model(pyfunc_path)
    y_pred = loaded_dynamic.predict(x_test)
    print('Prediction: \n', y_pred[:5])

    # custom model with predict_proba
    y_pred_proba = loaded_dynamic.predict(x_test, params={"predict_method": "predict_log_proba"})
    print('Predictive Probability: \n', y_pred_proba[:5])



