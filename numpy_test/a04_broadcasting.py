import numpy as np
from util import show


def main():
    a1 = np.linspace(1, 12, 12).reshape((3, 4))
    a2 = np.arange(1, 5)
    a3 = a2.reshape((4, 1))
    a4 = a1.reshape((3, 1, 4))
    b1 = a1 + a2
    b2 = a2 + a3
    b3 = a1 + a4
    show("a1:", a1) ; show("a2:", a2) ; show("a3:", a3)
    show("b1:", b1) ; show("b2:", b2) ; show("b3:", b3)

if __name__ == "__main__":
    main()
