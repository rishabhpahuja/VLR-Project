from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
import cv2
import time
import datetime
import os
import ipdb




# ********************************* Use this for Yolov7 
'''
1.Uncomment the below code
2.Change YOLOver=V7 in bridgewrapper.py 
'''
# !wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt


detector = Detector(classes = [0]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
detector.load_model('./yolov7x.pt')
video_path='./test_video/Stepladder_work.MP4'
YOLOVER='V7'


# ********************************* Use this for Yolov3
# video_path='./test_video/apples.mp4'
# detector=cv2.dnn.readNetFromDarknet("./yolo_weights/yolov3.cfg","./yolo_weights/yolov3_last.weights")
# YOLOVER='V3'



tracker=YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

# ipdb.set_trace()
date = datetime.datetime.now()
experiment_name = "Stepladder1_ds_newkf-sg_cos"
exp_directory = experiment_name #+ '_' + date.strftime("%m-%d_%H:%M:%S")
curr_dir=os.getcwd()+'/Tests/'
exp_directory=os.path.join('./Tests', exp_directory)
os.mkdir(exp_directory)


tracker.track_video(video_path, output="./IO_data/output/SG+IOU.avi",show_live =False, skip_frames = 0, count_objects = True, verbose=1,YOLOVER=YOLOVER,dir_path=exp_directory)
