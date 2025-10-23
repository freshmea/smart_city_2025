# pip install numpy
# pip install pandas

import numpy as np
import pandas as pd
from util import show


def main():
    # numpy 배열을 가지고 pandas Series 생성
    arr = np.arange(100, 131, dtype=np.uint8)
    show("Numpy Array:", arr)
    s1 = pd.Series(arr)
    print(s1)

    # list 로부터 pandas Series 생성
    s2 = pd.Series(['부장', '차장', '대리', '사원', '인턴'])
    print(s2)

if __name__ == "__main__":
    main()
