# 사용자 정의 타입 class
class MyClass:
    # class 정의 초기화함수
    def __init__(self, a : int, b) -> None:
        self.a = a
        self.b = b

    # 메소드 정의와 self 의 의미
    def method1(self, c):
        print("a:", self.a)
        print("b:", self.b)


def main():
    obj1 = MyClass(10, 20)
    # print(obj1.__class__)
    # print(obj1.__dir__())
    # print(obj1)
    print("a:", obj1.a)
    print("b:", obj1.b)
    # print("c:", obj1.c)
    obj1.method1(2)
    obj2 = MyClass(10, 20)
    MyClass(3, "test")
    print(obj2.a)

if __name__ == "__main__":
    main()
