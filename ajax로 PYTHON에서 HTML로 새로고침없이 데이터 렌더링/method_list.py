from shapely.geometry import Polygon
from shapely.geometry.point import Point

def big_hole(boxes):
    sqaure_polygon = None  # 먼저 변수를 정의합니다.

    if abs(boxes[0][0]-boxes[0][2]) < abs(boxes[1][0]-boxes[1][2]):
        bbox_center = (boxes[1][0] + boxes[1][2]) / 2, (boxes[1][1] + boxes[1][3]) / 2 
        # 정사각형의 꼭지점들을 계산
        square_vertices = [(bbox_center[0] - 3, bbox_center[1] - 3),  # 좌하단
                            (bbox_center[0] + 3, bbox_center[1] - 3),  # 우하단
                            (bbox_center[0] + 3, bbox_center[1] + 3),  # 우상단
                            (bbox_center[0] - 3, bbox_center[1] + 3)]  # 좌상단
        # Shapely의 Polygon 객체 생성
        sqaure_polygon = Polygon(Point(x, y) for x, y in square_vertices)
    else:
        bbox_center = (boxes[0][0] + boxes[0][2]) / 2, (boxes[0][1] + boxes[0][3]) / 2 
        # 정사각형의 꼭지점들을 계산
        square_vertices = [(bbox_center[0] - 5, bbox_center[1] - 5),  # 좌하단
                            (bbox_center[0] + 5, bbox_center[1] - 5),  # 우하단
                            (bbox_center[0] + 5, bbox_center[1] + 5),  # 우상단
                            (bbox_center[0] - 5, bbox_center[1] + 5)]  # 좌상단    
        # Shapely의 Polygon 객체 생성
        sqaure_polygon = Polygon(Point(x, y) for x, y in square_vertices)    
    
    return sqaure_polygon

def small_hole(boxes):
    sqaure_polygon = None  # 먼저 변수를 정의합니다.

    if abs(boxes[0][0]-boxes[0][2]) < abs(boxes[1][0]-boxes[1][2]):
        bbox_center = (boxes[0][0] + boxes[0][2]) / 2, (boxes[0][1] + boxes[0][3]) / 2 
        # 정사각형의 꼭지점들을 계산
        square_vertices = [(bbox_center[0] - 1, bbox_center[1] - 1),  # 좌하단
                            (bbox_center[0] + 1, bbox_center[1] - 1),  # 우하단
                            (bbox_center[0] + 1, bbox_center[1] + 1),  # 우상단
                            (bbox_center[0] - 1, bbox_center[1] + 1)]  # 좌상단
        # Shapely의 Polygon 객체 생성
        sqaure_polygon = Polygon(Point(x, y) for x, y in square_vertices)
    else:
        bbox_center = (boxes[1][0] + boxes[1][2]) / 2, (boxes[1][1] + boxes[1][3]) / 2 
        # 정사각형의 꼭지점들을 계산
        square_vertices = [(bbox_center[0] - 1, bbox_center[1] - 1),  # 좌하단
                            (bbox_center[0] + 1, bbox_center[1] - 1),  # 우하단
                            (bbox_center[0] + 1, bbox_center[1] + 1),  # 우상단
                            (bbox_center[0] - 1, bbox_center[1] + 1)]  # 좌상단    
        # Shapely의 Polygon 객체 생성
        sqaure_polygon = Polygon(Point(x, y) for x, y in square_vertices)    
    
    return sqaure_polygon

def sort(boxes):
    #오름차순으로 작은 홀, 큰 홀 boxes 크기 정렬
    if abs(boxes[0][0]-boxes[0][2]) < abs(boxes[1][0]-boxes[1][2]):     
        return boxes[0], boxes[1] 
    else:
        return boxes[1], boxes[0]
