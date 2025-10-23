import numpy as np
import pandas as pd


def main():
    s = pd.Series(['부장', '차장', '대리', '사원', '인턴'])
    print(s[0])
    # fancy indexing
    print(s[[1, 2, 3]]) # type: ignore
    print(s[np.arange(1, 4, 2)]) # type: ignore
    np.random.seed(0)
    s = pd.Series(np.random.randint(10000, 20000, size=(10,)))
    print(s)
    print( s > 15000)
    print( type(s > 15000))
    print(s[s>15000])  # boolean indexing

if __name__ == "__main__":
    main()
