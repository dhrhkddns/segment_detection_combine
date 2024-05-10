import threading
import time

# 스레드가 실행할 함수
def thread_function(name):
    print("스레드 '{}' 시작".format(name))
    time.sleep(2)  # 2초 동안 대기
    print("스레드 '{}' 종료".format(name))

if __name__ == "__main__":
    # 스레드 생성
    t1 = threading.Thread(target=thread_function, args=("첫 번째",))
    t2 = threading.Thread(target=thread_function, args=("두 번째",))

    # 스레드 시작
    t1.start()
    t2.start()

    # 메인 스레드에서도 작업을 수행할 수 있습니다.
    print("메인 스레드도 일을 합니다.")

    # 모든 스레드가 종료될 때까지 대기
    t1.join()
    t2.join()

    print("모든 스레드가 종료되었습니다.")
