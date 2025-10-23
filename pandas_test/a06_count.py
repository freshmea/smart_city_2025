# pip install seaborn

import numpy as np
import pandas as pd
import seaborn as sns

def main():
    df = sns.load_dataset("titanic")
    print(df.head())
    print(df.info())

    print(df['who'].value_counts())
    # 타입 변환
    print(df['pclass'].astype('int32').head())
    print(df['pclass'].astype('float32').head())
    print(df.info()) # 원본 데이터는 명시하지 않으면 바뀌지 않는다.
    # 정렬
    print(df.sort_values(by='age').head())
    print(df.sort_values(by='age', ascending=False, inplace=True))
    print(df.head()) # 원본 데이터는 명시하지 않으면 바뀌지 않는다.
    # 조건
    cond = (df['age'] >= 70)
    df.loc[cond]
    print(df.loc[cond][['sex', 'who', 'alive']])
    cond2 = (df['who'] == 'woman')
    cond1 = (df['fare'] > 30)
    print(df.loc[cond1 & cond2, ['sex', 'who', 'alive']])

if __name__ == "__main__":
    main()
