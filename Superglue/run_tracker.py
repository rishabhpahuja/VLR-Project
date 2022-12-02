from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
import cv2
import time
import datetime
import os
import ipdb

video_path='./test_video/apples.mp4'

net=cv2.dnn.readNetFromDarknet("./yolo_weights/yolov3.cfg","./yolo_weights/yolov3_last.weights")
tracker=YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=net)
ipdb.set_trace()
date = datetime.datetime.now()
experiment_name = "DeepSort"
exp_directory = experiment_name + '_' + date.strftime("%m-%d_%H:%M:%S")
curr_dir=os.getcwd()+'/Tests/'
os.mkdir(os.path.join('./Tests', exp_directory))


tracker.track_video(video_path, output="./IO_data/output/street_rp.avi",show_live =False, skip_frames = 0, count_objects = True, verbose=1,dir_path=exp_directory)
