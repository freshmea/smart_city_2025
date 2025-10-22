# pip install numpy
# pip install pandas

import numpy as np
import pandas as pd

def main():
    arr1 = np.array([1, 2, 3])
    print(arr1)
    print(type(arr1))

    df = pd.Series(arr1)
    print(df)
    print(type(df))

if __name__ == "__main__":
    main()
