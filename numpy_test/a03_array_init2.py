import numpy as np
from util import show


def main():
    a1 = np.arange(27+1, dtype=np.int8)
    a2 = np.arange(1, 11.000001, 0.5)
    a3 = np.linspace(1, 10+1, 20)
    show("a1:", a1)
    show("a2:", a2)
    show("a3:", a3)

if __name__ == "__main__":
    main()
