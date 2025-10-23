import pandas as pd
from util import pd_show


def main():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    pd_show("df", df)
    df2 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['가', '나', '다'])
    pd_show("df2", df2)

    data = {
    'name': ['Kim', 'Lee', 'Park'],
    'age': [24, 27, 34],
    'children': [2, 1, 3]
    }
    df3 = pd.DataFrame(data)
    pd_show("df3", df3)

if __name__ == "__main__":
    main()
