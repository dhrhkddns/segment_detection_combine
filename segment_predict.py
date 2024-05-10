from ultralytics import YOLO
import cv2
import numpy as np
from shapely.geometry import Polygon

# Load a pretrained YOLOv8n model
model_greenzone = YOLO(r'segment_100times_golf_best_openvino_model',task='segment')


# model.predict(source=r'C:\Users\ION\Desktop\daniel\GOLF-BALL-DETECTION-2-(seg)-2-2\valid\images',show_boxes=False,show_conf=False, save=True, imgsz=640, conf=0.5,classes=[1],line_width=1)

results = model_greenzone(source =1,classes=[1])

#골프 그린존의 꼭짓점 모음
#print(results[0].masks.xy[0])

print(results[0].masks.xy[0][0])

# 이미지 읽기
image = cv2.imread("golf.jpg")

# 다각형을 그릴 점들 정의 (예: 4개의 점)
points = np.array(results[0].masks.xy[0], np.int32)

# numpy 배열을 리스트로 변환
points_list = points.tolist()

# Shapely의 Polygon 객체 생성
polygon = Polygon(points_list)

# 결과 출력
print(polygon)
# 다각형 그리기
cv2.polylines(image, [points], True, (255, 0, 0), thickness=1)

# 다각형 내부를 파란색으로 채우기
#cv2.fillPoly(image, [points], (255, 0, 0))

# 이미지 보여주기
cv2.imshow("Polygon", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 결과 이미지를 파일로 저장하려면 다음 코드를 사용합니다.
cv2.imwrite("output_golf.jpg", image)
