import numpy as np
import pandas as pd


def show(name : str, arr : np.ndarray):
    print(f"{name} :\n{arr}, shape: {arr.shape}, dtype: {arr.dtype}")

def pd_show(name : str, arr : pd.DataFrame):
    print(f"{name} :\n{arr}, shape: {arr.shape}, dtype: \n{arr.dtypes}")