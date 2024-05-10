# Ultralytics YOLO 🚀, AGPL-3.0 license
from collections import defaultdict
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

vid_frame_count = 0  # 전역 변수로 정의


track_history = defaultdict(list)
track_lst = []

current_region = None
counting_regions = [
    {
         'name': 'small_hall',
         'polygon': Polygon([(75.0, 325.0), (74.92314112161293, 324.2196387119355), (74.69551813004514, 323.46926627053966), (74.32587844921018, 322.77771906792157), (73.82842712474618, 322.1715728752538), (73.2222809320784, 321.67412155078983), (72.53073372946037, 321.3044818699548), (71.78036128806451, 321.0768588783871), (71.0, 321.0), (70.21963871193549, 321.0768588783871), (69.46926627053963, 321.3044818699548), (68.7777190679216, 321.67412155078983), (68.17157287525382, 322.1715728752538), (67.67412155078982, 322.77771906792157), (67.30448186995486, 323.46926627053966), (67.07685887838707, 324.2196387119355), (67.0, 325.0), (67.07685887838707, 325.7803612880645), (67.30448186995486, 326.53073372946034), (67.67412155078982, 327.22228093207843), (68.17157287525382, 327.8284271247462), (68.7777190679216, 328.32587844921017), (69.46926627053965, 328.6955181300452), (70.21963871193549, 328.9231411216129), (71.0, 329.0), (71.78036128806451, 328.9231411216129), (72.53073372946037, 328.6955181300452), (73.2222809320784, 328.32587844921017), (73.82842712474618, 327.8284271247462), (74.32587844921018, 327.22228093207843), (74.69551813004514, 326.53073372946034), (74.92314112161291, 325.7803612880645)]),  # Polygon points
         'counts': 0,
         'dragging': False,
         'region_color': (6, 97, 255),  # BGR Value
         'text_color': (255, 255, 255),  # Region Text Color
         'in' : False
     },
    {
        'name': 'big_hall',
        'polygon': Polygon([(71.0, 362.0), (70.86549696282262, 360.6343677458871), (70.46715672757901, 359.3212159734444), (69.82028728611782, 358.11100836886277), (68.94974746830583, 357.0502525316942), (67.88899163113722, 356.17971271388217), (66.67878402655563, 355.532843272421), (65.3656322541129, 355.1345030371774), (64.0, 355.0), (62.634367745887104, 355.1345030371774), (61.321215973444374, 355.532843272421), (60.111008368862784, 356.17971271388217), (59.05025253169417, 357.0502525316942), (58.17971271388218, 358.11100836886277), (57.53284327242099, 359.3212159734444), (57.13450303717739, 360.6343677458871), (57.0, 362.0), (57.13450303717739, 363.3656322541129), (57.53284327242099, 364.6787840265556), (58.17971271388218, 365.88899163113723), (59.05025253169417, 366.9497474683058), (60.11100836886278, 367.82028728611783), (61.321215973444374, 368.467156727579), (62.6343677458871, 368.8654969628226), (64.0, 369.0), (65.3656322541129, 368.8654969628226), (66.67878402655563, 368.467156727579), (67.88899163113722, 367.82028728611783), (68.94974746830583, 366.9497474683058), (69.82028728611782, 365.88899163113723), (70.46715672757901, 364.6787840265556), (70.86549696282262, 363.3656322541129)]),  # Polygon points
        'counts': 0,
        'dragging': False,
        'region_color': (37, 255, 225),  # BGR Value
        'text_color': (0, 0, 0),  # Region Text Color
        'in' : False
    },
    {
         'name': 'green_hall',
         'polygon': Polygon([(43, 386), (45, 323), (51, 307), (62, 304), (638, 310), (638, 384)]),  # Polygon points
         'counts': 0,
         'dragging': False,
         'region_color': (255, 42, 4),  # BGR Value 파란색
         'text_color': (255, 255, 255),  # Region Text Color
         'in' : False
    },
    {
         'name': 'start_region',
         'polygon': Polygon([(517, 308), (541, 308), (541, 384), (517, 384)]),  # Polygon points
         'counts': 0,
         'dragging': False,
         'region_color': (0, 0, 0),  # BGR Value 검은색
         'text_color': (255, 255, 255),  # Region Text Color
         'in' : False
    } ]





def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse events for region manipulation.

    Parameters:
        event (int): The mouse event type (e.g., cv2.EVENT_LBUTTONDOWN).
        x (int): The x-coordinate of the mouse pointer.
        y (int): The y-coordinate of the mouse pointer.
        flags (int): Additional flags passed by OpenCV.
        param: Additional parameters passed to the callback (not used in this function).

    Global Variables:
        current_region (dict): A dictionary representing the current selected region.

    Mouse Events:
        - LBUTTONDOWN: Initiates dragging for the region containing the clicked point.
        - MOUSEMOVE: Moves the selected region if dragging is active.
        - LBUTTONUP: Ends dragging for the selected region.

    Notes:
        - This function is intended to be used as a callback for OpenCV mouse events.
        - Requires the existence of the 'counting_regions' list and the 'Polygon' class.

    Example:
        >>> cv2.setMouseCallback(window_name, mouse_callback)
    """
    global current_region

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def run(
    weights="C:\\Users\\omyra\\OneDrive\\바탕 화면\\again\\gwangun_and_golf_best_openvino_model",
    source=1,
    device="cpu",
    view_img=True,
    save_img=False,
    exist_ok=False,
    classes=0,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
):
    """
    Run Region counting on a video using YOLOv8 and ByteTrack.

    Supports movable region for real time counting inside specific area.
    Supports multiple regions counting.
    Regions can be Polygons or rectangle in shape

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        device (str): processing device cpu, 0, 1
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        classes (list): classes to detect and track_lst
        line_thickness (int): Bounding box thickness.
        track_thickness (int): Tracking line thickness
        region_thickness (int): Region thickness.
    """

    
    global vid_frame_count, ready, 굴러가는_상태, 공_움직임,공_홀_안에_들어감,골_그린존_바깥에_나감, score, 판단_해야하는_키
    
    score = 0
    ready = False
    굴러가는_상태 = False
    공_움직임 = False
    공_홀_안에_들어감 = False
    골_그린존_바깥에_나감 = False

    판단_해야하는_키 = False
    # Check source path
    # if not Path(source).exists():
    #     raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(f"{weights}")
    # model.to("cuda") if device == "0" else model.to("cpu")

    model_greenzone = YOLO(r'segment_100times_golf_best_openvino_model',task='segment')

    # Extract classes names
    # names = model.model.names

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    # save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
    # save_dir.mkdir(parents=True, exist_ok=True)
    # video_writer = cv2.VideoWriter(str(save_dir / "god.mp4"), fourcc, fps, (frame_width, frame_height))

    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        #frame = cv2.flip(frame, 1) #0은 상하 1은 좌우 반전

        if not success:
            break
        vid_frame_count += 1

        #print("현재 프레임 : " + str(vid_frame_count))

        results_greenzone = model_greenzone(frame , classes=[1],conf=0.7)

        if results_greenzone[0].masks is not None:
            print(results_greenzone[0].masks.xy[0] )
            counting_regions[2]['polygon']= results_greenzone[0].masks.xy[0] 

        # 이거는 golf_ball
        results = model.track(frame, persist=True, classes=0, conf=0.5,verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # print(f"track_ids: {track_ids}")
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str("golf_ball"))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                #annotator.box_label(box, str("GOLF_BALL"), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                #track = track_history[track_id]  # Tracking Lines plot
                track_lst.append((float(bbox_center[0]), float(bbox_center[1])))


                if len(track_lst) > 30:
                    track_lst.pop(0)
                points = np.hstack(track_lst).astype(np.int32).reshape((-1, 1, 2))
                #cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                if len(track_lst) == 30:
                    distance = sum(((track_lst[i][0] - track_lst[i+1][0]) ** 2 + (track_lst[i][1] - track_lst[i+1][1]) ** 2) ** 0.5 for i in range(len(track_lst)-1))
                    #print(int(distance))
                    if distance < 15:
                        공_움직임 = False
                    else:
                        공_움직임 = True

                #게임은 시작했는데 아직 준비 상태가 아닐때, 영역에 놓음으로써 준비상태로 만들어주는 코드        
                if ready == False: 
                    if counting_regions[3]['polygon'].contains(Point((bbox_center[0], bbox_center[1]))):
                        #print("시작영역에 공이 들어옴")
                        counting_regions[3]['in'] = True
                        if len(track_lst) == 30:
                            #print("30프레임 이상 공이 들어옴")
                            distance = sum(((track_lst[i][0] - track_lst[i+1][0]) ** 2 + (track_lst[i][1] - track_lst[i+1][1]) ** 2) ** 0.5 for i in range(len(track_lst)-1))
                            #distance = ((track_lst[0][0] - track_lst[-1][0]) ** 2 + (track_lst[0][1] - track_lst[-1][1]) ** 2) ** 0.5
                            if distance < 15:
                                ready = True
                                print("하하하하하하 공들어 왔다")
                                #print(int(distance))
                                #print("공이 영역안에서 멈춤")
                
                if ready == True:
                   
                    if 공_움직임 == True: 
                        print("공 움직이는 중")
                        굴러가는_상태 = True
                        

                        ready = False #공이 굴러가면 준비상태가 아니게 됨
                
                if 굴러가는_상태 == True:
                    
                    if counting_regions[0]['polygon'].contains(Point((bbox_center[0], bbox_center[1]))) or counting_regions[1]['polygon'].contains(Point((bbox_center[0], bbox_center[1]))):
                        공_홀_안에_들어감 = True
                        #굴러가는_상태 = False
                    
                    elif counting_regions[2]['polygon'].contains(Point((bbox_center[0], bbox_center[1]))) == False:
                        #score += 1
                        #print(f"나감 {score}")
                        골_그린존_바깥에_나감 = True
                        #굴러가는_상태 = False
                    
                    elif 공_움직임 == False: #굴러가다가 그린존에서 멈췄을때
                        #score += 1
                        #print(f"공 멈춤 {score}")
                        #굴러가는_상태 = False   
                        ready = True    

                # Check if detection inside region
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            # cv2.rectangle(
            #     frame,
            #     (text_x - 5, text_y - text_size[1] - 5),
            #     (text_x + text_size[0] + 5, text_y + 5),
            #     region_color,
            #     -1,
            # )
            # q
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=1)

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
                cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", mouse_callback)
            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)


        for region in counting_regions:  # Reinitialize count for each region
            region["counts"] = 0
            region["in"] = False

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count

    videocapture.release()
    cv2.destroyAllWindows()

def game_condition():
    
    global ready, 굴러가는_상태, 공_움직임,공_홀_안에_들어감,골_그린존_바깥에_나감, score, 판단_해야하는_키
    
    score_list = []

    num = 0

    while len(score_list) <= 2:
        

        while True:
            ready = False
            굴러가는_상태 = False
            공_움직임 = False
            공_홀_안에_들어감 = False
            골_그린존_바깥에_나감 = False

            num += 1
            with open("shared_value.txt", "w") as f:
                f.write(str(num))

            a,b,c = 0,0,0
            time.sleep(0.5)
            print(f"{len(score_list)+1}번째 홀, 게임 시작 상태")
            print("-------------------")
            while ready == True:
                time.sleep(0.1)
                print("준비 상태")
                골_그린존_바깥에_나감 = False

                while 굴러가는_상태 == True:
                    time.sleep(0.1)
                    # print("굴러가는 상태")

                    if 공_홀_안에_들어감 == True:
                        print(f"공이 홀 안에 들어감, 이번 홀 점수는 {score}점 입니다.")
                        score_list.append(score)

                        score = 0 #다시 새로운 홀이니까 점수 0으로 초기화
                        a += 1
                        b += 1

                        굴러가는_상태 = False
                        
                        break

                    elif 골_그린존_바깥에_나감 == True:
                        print("골 그린존 바깥에 나감")
                        score += 1
                        print(f"나감 {score}")

                        굴러가는_상태 = False

                        a += 1
                        break
                    
                    elif 공_움직임 == False:
                        # print("공이 그린존 안에서 멈춤")
                        score += 1
                        print(f"공 멈춤 {score}")

                        굴러가는_상태 = False

                        break

                if a == 1:
                    #print("게임 시작 상태로 돌아감")
                    break
            
            if b == 1:
                #print("게임 끝 상태로 돌아감")
                break
        
    print("게임 끝 전체 홀 점수는..." + str(score_list))





                    
                    


        


def main():
    """Main function."""
    # 스레드 생성
    thread1 = threading.Thread(target=game_condition)
    thread1.start()
    run()


if __name__ == "__main__":
    print("메인함수실행")
    main()
