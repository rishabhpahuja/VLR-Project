from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
import cv2

video_path='/home/saharsh2/VLR-Project/deep_sort/test_video/apples.mp4'

net=cv2.dnn.readNetFromDarknet("/home/saharsh2/VLR-Project/deep_sort/yolo_weights/yolov3.cfg","/home/saharsh2/VLR-Project/deep_sort/yolo_weights/yolov3_last.weights")
tracker=YOLOv7_DeepSORT(reID_model_path="/home/saharsh2/VLR-Project/deep_sort/deep_sort/model_weights/mars-small128.pb", detector=net)

tracker.track_video(video_path, output="./IO_data/output/street_rp.avi",show_live =False, skip_frames = 0, count_objects = True, verbose=1,dir_path='/home/saharsh2/VLR-Project/deep_sort/deep_sort/Tests/')
