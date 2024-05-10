
from roboflow import Roboflow
rf = Roboflow(api_key="IOJEJZBxTKUuevke1feJ")
project = rf.workspace("appleroot-zk4px").project("golf-ball-detection-2-seg")
version = project.version(9)
dataset = version.download("yolov8")

