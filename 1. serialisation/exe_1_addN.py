"""
----------------------------------------------------------------------------------------
addN.py

How to serialise a model using MLflow's PythonModel API.

Source:
-------
    : https://mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/notebooks/introduction.html

: zach.wolpe@medibio.com.au
: 30-07-24
----------------------------------------------------------------------------------------
"""

import mlflow.pyfunc
import pandas as pd
import logging


class AddN(mlflow.pyfunc.PythonModel):
    """
    A custom model that adds a specified value `n` to all columns of the input DataFrame.

    Attributes:
    -----------
    n : int
        The value to add to input columns.
    """

    def __init__(self, n):
        """
        Constructor method. Initializes the model with the specified value `n`.

        Parameters:
        -----------
        n : int
            The value to add to input columns.
        """
        self.n = n

    def predict(self, context, model_input, params=None):
        """
        Prediction method for the custom model.

        Parameters:
        -----------
        context : Any
            Ignored in this example. It's a placeholder for additional data or utility methods.

        model_input : pd.DataFrame
            The input DataFrame to which `n` should be added.

        params : dict, optional
            Additional prediction parameters. Ignored in this example.

        Returns:
        --------
        pd.DataFrame
            The input DataFrame with `n` added to all columns.
        """
        return model_input.apply(lambda column: column + self.n)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # Define the path to save the model
    model_path = "./tmp/add_n_model"

    # Create an instance of the model with `n=5`
    logging.info("Creating a model with n=5.")
    add5_model = AddN(n=5)

    # Save the model using MLflow
    logging.info("Saving the model to %s." % model_path)
    mlflow.pyfunc.save_model(path=model_path, python_model=add5_model)

    logging.info("Model saved successfully!")

    # Load the saved model
    logging.info("Loading the model from %s." % model_path)
    loaded_model = mlflow.pyfunc.load_model(model_path)

    # Define a sample input DataFrame
    model_input = pd.DataFrame([range(10)])

    # Use the loaded model to make predictions
    logging.info("Making predictions with the loaded model.")
    model_output = loaded_model.predict(model_input)

    _output = "The model output is: \n{}".format(model_output)
    logging.info(_output)

    logging.info("Exited with status_code=1. Model loaded and predictions made successfully!")
    logging.info("Runtime Complete.")


    



    