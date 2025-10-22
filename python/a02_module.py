import a01_type

from test_package import a_module
from test_package import b_module

def main():
    print(a_module.a_var)
    print(b_module.b_var)

if __name__ == "__main__":
    main()
