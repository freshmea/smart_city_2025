import numpy as np
import pandas as pd


def main():
    s = pd.Series(['마케팅', '경영', '개발', '기획', '인사'], index=['a', 'b', 'c', 'd', 'e'])
    print(s['c'])
    print(s[['a', 'd']])
    s2 = pd.Series(['마케팅', '경영', '개발', '기획', '인사'])
    print(s2)
    s2.index = list('abcde') # type: ignore
    print(s2)

if __name__ == "__main__":
    main()
