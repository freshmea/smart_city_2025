import numpy as np
from util import show


def main():
    a1 = np.array([1, 2, 3, 4])
    a2 = np.array([[1, 2, 3], [4, 5, 6]])
    a3 = np.linspace(10, 120, 12).reshape((3, 2, 2))
    b1 = a1.T
    b1 = a1.transpose()
    b2 = a2.transpose()
    b3 = np.transpose(a3)
    show("a1:", a1) ; show("a2:", a2) ; show("a3:", a3)
    show("b1:", b1) ; show("b2:", b2) ; show("b3:", b3)

if __name__ == "__main__":
    main()
