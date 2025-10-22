import numpy as np


def show(name : str, arr : np.ndarray):
    print(f"{name} : {arr}, shape: {arr.shape}, dtype: {arr.dtype}")

def main():
    a1 = np.array([1, 2, 3], dtype=np.int8)
    a2 = np.array([1, 1.5, 2, 3])
    a3 = np.array([[1, 2, 3], [4, 5, 6]])

    show("a1", a1)
    show("a2", a2)
    show("a3", a3)

if __name__ == "__main__":
    main()
