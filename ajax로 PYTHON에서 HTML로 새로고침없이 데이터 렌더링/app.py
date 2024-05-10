from flask import Flask, request, jsonify,render_template
import random
import threading
from colorama import init, Fore
#작업 디렉토리가 달라서 설정해주고 실시간으로 변하는 변수를 다른 py파일에서 받아온다.


# colorama 초기화
init(autoreset=True)


app = Flask(__name__)
@app.route('/data', methods=['GET'])
def get_data():
    data = {'name': 'John', 'age': 30, 'city': 'New York'}
    return jsonify(data)  # 'age' 키에 대한 값만 반환

@app.route('/', methods=['GET'])
def home():
    return render_template('golf.html')

@app.route('/gwangun', methods=['GET'])
def yoooo():
    data = {'name': 'John', 'age': 30, 'city': 'New York'}
    print(Fore.BLUE +"python return 했음")
    return str(random.randint(1, 10))
    

@app.route('/golf_score', methods=['GET'])
def golf_score():
    with open("shared_value.txt", "r") as a:
        content = int(a.read())
    return str(content) 

@app.route('/animation_complete', methods=['POST'])
def process_data():
    data = request.json
    received_data = data.get('data')
    
    with open("shared_value.txt", "w") as a:
        a.write(str(received_data))

    return jsonify({'message': 'Data received successfully'})

if __name__ == '__main__':
    #스레드 생성
    t1 = threading.Thread(target=app.run)

    #스레드 시작
    t1.start()


print(Fore.BLUE + "wanna do BOTH AT THE SAME TIMEEEE!!")

#이 py를 영상인식 골프 py 결국에는 합쳐야한다....ㄷㄷ (홀에 들어간 스코어 변수를 html에 보여줘야함)