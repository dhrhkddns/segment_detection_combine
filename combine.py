from ultralytics import YOLO
import cv2
import numpy as np

# 이미지 읽기
image = cv2.imread("golf.jpg")

#detect 골프공, 홀 segment 그린존 
model_detect = YOLO(r'detect_300times_golf_best_openvino_model',task='detect')
model_segment = YOLO(r'segment_100times_golf_best_openvino_model', task='segment')

results_ball = model_detect(source=r'golf.jpg',classes=[0])
results_holes = model_detect(source=r'golf.jpg',classes=[2],conf=0.6)
results_green = model_segment(source =r'golf.jpg',classes=[1])

#그린존, 공, 검은홀의 좌표 
coordinates_ball = results_ball[0].boxes.xyxy
coordinates_holes = results_holes[0].boxes.xyxy
coordinates_green = results_green[0].masks.xy

#공, 검은홀은 소수점이여서 polyline으로 표현하기 위해 int로 바꿔준다.
coordinates_ball = np.array(coordinates_ball, np.int32)
coordinates_holes = np.array(coordinates_holes, np.int32)
coordinates_green = np.array(coordinates_green, np.int32)

#홀은 2개니까 나누기
first_hole = coordinates_holes[0]
second_hole = coordinates_holes[1]

#공과 검은홀의 중심 구하기
#4개의 정수 꼭짓점에서 공의 x중심 y중심 구하기
center_x = int((coordinates_ball[0][0] + coordinates_ball[0][2]) / 2)
center_y = int((coordinates_ball[0][1] + coordinates_ball[0][3]) / 2)

first_hole_center = (int((first_hole[0] + first_hole[2]) / 2) , int((first_hole[1] + first_hole[3]) / 2))
second_hole_center = (int((second_hole[0] + second_hole[2]) / 2) , int((second_hole[1] + second_hole[3]) / 2))

# 공과 홀들은 원 그리기, 그린존은 다각형 영역 그리기
cv2.circle(image, (center_x,center_y), radius=2, color=(0, 0, 255), thickness=1)  
cv2.circle(image, (first_hole_center), radius=6, color=(0, 0, 255), thickness=1) 
cv2.circle(image, (second_hole_center), radius=4, color=(0, 0, 255), thickness=1) 
cv2.polylines(image, [coordinates_green], True, (255, 0, 0), thickness=1)

# # 이미지 보여주기
cv2.imshow("Polygon", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
