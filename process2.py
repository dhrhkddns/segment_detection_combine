import time
from multiprocessing import Process, Value

# 공유 변수 정의
shared_variable = Value('i', 0)

# 공유 변수를 변경하는 함수
def update_shared_variable():
    while True:
        with shared_variable.get_lock():
            shared_variable.value += 1
            print("두 번째 프로세스에서 공유 변수 값:", shared_variable.value)
        time.sleep(1)

# 공유 변수 변경 함수 실행
update_shared_variable()
