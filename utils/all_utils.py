import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from matplotlib.colors import ListedColormap
import os

plt.style.use("fivethirtyeight")
    """it is used to sepeate the dependent and independent variables
    """
def prepare_data(df):
    """it is used to sepeate the dependent and independent variables

    Args:
        df (pd.DataFame): _description_

    Returns:
        _type_: _description_
    """
    X = df.drop("y", axis = 1)
    y = df["y"]
    return X, y

    ##SAVING THE MODESL
def save_model(model, filename):
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok = True)
    filePath = os.path.join(model_dir, filename)
    joblib.dump(model, filePath)