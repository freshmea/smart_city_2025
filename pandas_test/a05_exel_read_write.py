import numpy as np
import pandas as pd


def main():
    excel = pd.read_excel('data/scientists_edit_ed.xlsx', sheet_name='sci_edit', engine='openpyxl')
    print(excel.head())
    excel.to_excel('data/sample1.xlsx', index=False, sheet_name='샘플')

if __name__ == "__main__":
    main()
