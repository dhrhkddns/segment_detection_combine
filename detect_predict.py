from ultralytics import YOLO
import cv2
import numpy as np
# Load a pretrained YOLOv8n model
model = YOLO(r'detect_300times_golf_best_openvino_model')


results_ball = model(source=r'golf.jpg',classes=[0])

results_holes = model(source=r'golf.jpg',classes=[2])

# 이미지 읽기
image = cv2.imread("golf.jpg")

#텐서 요소의 꼭짓점들
points = results_ball[0].boxes.xyxy

holes_points = results_holes[0].boxes.xyxy



#텐서 모든 요소를 정수로 변환 polyline 작성 하기 위해서
int_points = np.array(points, np.int32)

#검은 홀들의 정수 좌표 2차원 리스트
int_holes_points = np.array(holes_points, np.int32)

first_black_hole = int_holes_points[0]

second_black_hole = int_holes_points[1]

#4개의 정수 꼭짓점에서 공의 x중심 y중심 구하기
center_x = int((int_points[0][0] + int_points[0][2]) / 2)
center_y = int((int_points[0][1] + int_points[0][3]) / 2)

first_black_hole_points = (int((first_black_hole[0] + first_black_hole[2]) / 2) , int((first_black_hole[1] + first_black_hole[3]) / 2))
second_black_hole_points = (int((second_black_hole[0] + second_black_hole[2]) / 2) , int((second_black_hole[1] + second_black_hole[3]) / 2))


# 점 그리기
cv2.circle(image, (center_x,center_y), radius=2, color=(0, 0, 255), thickness=1)  # 두께를 -1로 설정하여 내부를 채움
cv2.circle(image, (first_black_hole_points), radius=2, color=(255, 0, 0), thickness=1)  # 두께를 -1로 설정하여 내부를 채움
cv2.circle(image, (second_black_hole_points), radius=2, color=(255, 0, 0), thickness=1)  # 두께를 -1로 설정하여 내부를 채움

# 이미지 보여주기
cv2.imshow("golf_ball", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


