"""
----------------------------------------------------------------------------------------
Lissajous.py

How to serialise a model using MLflow's PythonModel API, with the following additions:

    - Arbitrary return types (plot).
    - The separation of the model's parameters and the input data.
    - The introduction of model signatures.


: zach.wolpe@medibio.com.au
: 30-07-24
----------------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import os
import shutil

from mlflow.models import infer_signature
import mlflow.pyfunc
import warnings

warnings.filterwarnings("ignore")


class Lissajous(mlflow.pyfunc.PythonModel):
    def __init__(self, A=1, B=1, num_points=1000):
        self.A = A
        self.B = B
        self.num_points = num_points
        self.t_range = (0, 2 * np.pi)

    def generate_lissajous(self, a, b, delta):
        t = np.linspace(self.t_range[0], self.t_range[1], self.num_points)
        x = self.A * np.sin(a * t + delta)
        y = self.B * np.sin(b * t)
        return pd.DataFrame({"x": x, "y": y})

    def predict(self, context, model_input, params=None):
        """
        Generate and plot the Lissajous curve with annotations for parameters.

        Args:
        - model_input (pd.DataFrame): DataFrame containing columns 'a' and 'b'.
        - params (dict, optional): Dictionary containing optional parameter 'delta'.
        """
        # Extract a and b values from the input DataFrame
        a = model_input["a"].iloc[0]
        b = model_input["b"].iloc[0]

        # Extract delta from params or set it to 0 if not provided
        delta = params.get("delta", 0)

        # Generate the Lissajous curve data
        df = self.generate_lissajous(a, b, delta)

        sns.set_theme()

        # Create the plot components
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(df["x"], df["y"])
        ax.set_title("Lissajous Curve")

        # Define the annotation string
        annotation_text = f"""
        A = {self.A}
        B = {self.B}
        a = {a}
        b = {b}
        delta = {np.round(delta, 2)} rad
        """

        # Add the annotation with a bounding box outside the plot area
        ax.annotate(
            annotation_text,
            xy=(1.05, 0.5),
            xycoords="axes fraction",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "aliceblue", "edgecolor": "black"},
        )

        # Adjust plot borders to make space for the annotation
        plt.subplots_adjust(right=0.65)
        plt.show()
        plt.close()


        # Return the plot
        return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Define the path to save the model
    logging.info("Instantiating Lissajous model.")
    model_path = "./tmp/lissajous_model"

    # rm files from previous runs
    if os.path.exists(model_path):
        shutil.rmtree(model_path, ignore_errors=True)
        
    # Create an instance of the model, overriding the default instance variables `A`, `B`, and `num_points`
    model_10k_standard = Lissajous(1, 1, 10_000)

    # Infer the model signature, ensuring that we define the params that will be available for customization at inference time
    signature = infer_signature(
        model_input=pd.DataFrame([{"a": 1, "b": 2}]),
        params={"delta": np.pi / 5})

    # Save our custom model to the path we defined, with the signature that we declared
    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=model_10k_standard,
        signature=signature)
    logging.info("Model saved successfully!")

    # Load our custom model from the local artifact store
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_path)

    # Inference
    # Define the input DataFrame. In our custom model, we're reading only the first row of data to generate a plot.
    model_input = pd.DataFrame({"a": [3], "b": [2]})

    # Define a params override for the `delta` parameter
    params = {"delta": np.pi / 3}

    # Run predict, which will call our internal method `generate_lissajous` before generating a `matplotlib` plot showing the curve
    fig = loaded_pyfunc_model.predict(model_input, params)

    logging.info('Runtime Complete.')