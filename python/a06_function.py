def a_function(a, b, c):
    # a = 1
    # b = 2
    # c = 3
    print("A 함수입니다.", a ,b, c)
    return a + b + c, 2

def main():
    a, b = a_function(1, 20, 3)
    print("리턴값:", a, b)

if __name__ == "__main__":
    main()
