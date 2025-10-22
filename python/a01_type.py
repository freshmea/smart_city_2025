
def main():
    # 파이썬의 기본 타입
    print("안녕하세요.")
    print("방갑습니다.")

    a_var = 123
    print(a_var)
    print(type(a_var))
    a_var = "문자열도 가능합니다."
    print(a_var)
    print(type(a_var)) # 동적 타이핑

    # 파이썬의 타입은 primitive type 이 존재하지 않는다.
    # 오직 파이썬에는 모든 변수가 클래스의 객체이다.

    # 함수! 키워드

    # 함수의 등록!
    # 들여쓰기
    def a_func():
        print("안녕하세요.")
        print("방갑습니다.")

    # 함수를 불러온다.
    a_func()
    a_func()
    a_func()
    a_func()
    a_func()

if __name__ == "__main__":
    main()
