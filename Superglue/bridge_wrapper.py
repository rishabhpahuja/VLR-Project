'''
A Moduele which binds Yolov7 repo with Deepsort with modifications
'''
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow logging outputs
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder
from detection_helpers import *
import ipdb


 # load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True

prev_frame = None
class YOLOv7_DeepSORT:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''
    #done
    def __init__(self, reID_model_path:str, detector, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0,
    coco_names_path:str ="./io_data/input/classes/coco.names",  ):
        '''
        args: 
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        '''
        self.detector = detector
        self.coco_names_path = ['apples']
        self.nms_max_overlap = nms_max_overlap
        # self.class_names = read_class_names()

        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1) # becomes the encoder 
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric) # initialize tracker


    def track_video(self,video:str, output:str, skip_frames:int=0, show_live:bool=False, count_objects:bool=False, verbose:int = 0, YOLOVER='V3',dir_path='./', plt_unmatched_det= False):
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            count_objects: count objects being tracked on screen
            verbose: print details on the screen allowed values 0,1,2
        '''
        try: # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        out = None
        if output: # get video ready to save locally if flag is set
            # print(output) # output folder name and location - ./IO_data/output/street_rp.avi
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int # 2048
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 1536
            fps = int(vid.get(cv2.CAP_PROP_FPS)) # 15
            codec = cv2.VideoWriter_fourcc(*"XVID") # codec 
            out = cv2.VideoWriter(output, codec, fps, (width, height)) # writer ka object
            # print(width, height, fps, codec, out)

        frame_num = 0
        prev_frame = None
        while True: # while video is running
            return_value, frame = vid.read()
            # #ipdb.set_trace()
            frame_copy = copy.deepcopy(frame)
            if not return_value:
                print('Video has ended or failed!')
                break
            frame_num +=1
            # #ipdb.set_trace()

            if skip_frames and not frame_num % skip_frames: 
                continue # skip every nth frame. When every frame is not important, you can use this to fasten the process
            if verbose >= 1:
                start_time = time.time()
            

            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            if YOLOVER=='V3': # using V3 only 
                yolo_dets, scores, class_ID=YOLOV3(self.detector,frame)
                count = len(class_ID) # number of detections - 7 for frame 1
                #Class number. It will be zero for our case since there is only one class
                # obviously all 7 are apple
                names=np.array(["Apple"]*count)
                # print(yolo_dets)
                # first frame - [[1717, 680, 90, 77], [957, 1072, 79, 92], [1656, 326, 90, 73], [753, 1114, 79, 94], [1219, 231, 96, 76], [815, 471, 77, 76], [1516, 881, 79, 82]]
                # bounding boxes 

                # print(scores)
                # confidence level - [0.9174734354019165, 0.8873069882392883, 0.8572289943695068, 0.8180209994316101, 0.7712001800537109, 0.7203170657157898, 0.7065609693527222]
                # looks like ascending order

                # print(class_ID) # [0, 0, 0, 0, 0, 0, 0] - apple only 
            
            else:
                
                yolo_detects = self.detector.detect(frame.copy(), plot_bb = False)  # Get the detections
                if yolo_detects is None:
                    yolo_dets = []
                    scores = []
                    class_ID = []
                    num_objects = 0
                    count=0
                    names=np.array(["P"]*count)
            
                else:
                    # import ipdb; ipdb.set_trace()
                    bboxes = yolo_detects[:,:4]
                    bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
                    bboxes[:,3] = bboxes[:,3] - bboxes[:,1]
                    yolo_dets=(bboxes).tolist()

                    scores =( yolo_detects[:,4]).tolist()
                    class_ID =( yolo_detects[:,-1]).tolist()
                    count = len(class_ID)
                    names=np.array(["P"]*count)
                    num_objects = bboxes.shape[0]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
            
            
            


            if count_objects: # has been made true from run_tracker.py
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255), 2)
                # dislays top-left mein - number of objects being being tracked in the current frame
                # is it detections or tracks ?? Ask Pahuja - seems like detections only \
            

            # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
            # #ipdb.set_trace()
            features = self.encoder(frame, yolo_dets) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size] - 7,128 
            # features is after the bounding boxes go through the deepsort feature extracting network
            
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(yolo_dets, scores, names, features)] 
            # [No of BB per frame] deep_sort.detection.Detection object - len = 7
            # each detection is then taken individually and zipped with related values like scre, name and features.

            cmap = plt.get_cmap('tab20b') #initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]   # can't detect more than 20 at once ??
            # each has 3 values
            # #ipdb.set_trace()
            self.tracker.predict(prev_frame, frame)  # Call the tracker - nothing with the cost yet
            # #ipdb.set_trace()
            unmatched_tracks,unmatched_detections=self.tracker.update(detections) #  update using Kalman Gain
            # update - step 12 in my notebook
            import ipdb; #ipdb.set_trace()
            
            if plt_unmatched_det:
                for ud in unmatched_detections:
                    # print(ud)
                    tlbr_det = ud.to_tlbr()
                    cv2.rectangle(frame,(int(tlbr_det[0]), int(tlbr_det[1])), (int(tlbr_det[2]), int(tlbr_det[3])), (255,0,0), 5)

            for track in self.tracker.tracks:  # update new findings AKA tracks  
                # if not track.is_confirmed() or track.time_since_update > 1:
                #     continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()

                # color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                # color = [i * 255 for i in color]
                color = (0,255,0) # changed code 
                text_color=(255,255,255)
                if track.time_since_update > 0:
                    color=(255,255,255)
                    text_color=(0,0,0)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*20, int(bbox[1])), color, -1) #To make a solid rectangle box to write text on
                cv2.putText(frame, class_name + ":" + str(track.track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.8, (text_color),2, lineType=cv2.LINE_AA)  
                cv2.putText(frame, "Frame_num:"+str(frame_num),(len(frame[0])-300,len(frame)-100),0, 1.2, (255,255,255),2, lineType=cv2.LINE_AA)  
                if verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                    
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            # for track in unmatched_tracks:
            # #ipdb.set_trace()
            if verbose >= 1:
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count}")
            
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if output: out.write(result) # save output video

            if show_live:
                cv2.namedWindow("Output Video",cv2.WINDOW_NORMAL)
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            # if frame_num>9 and frame_num<100:
            #     name=dir_path+'00'+str(frame_num)+'.png'
            # elif frame_num<10:
            #     name=dir_path+'000'+str(frame_num)+'.png'
            # elif frame_num >99 and frame_num < 1000:
            #      name=dir_path+'0'+str(frame_num)+'.png'
            # elif frame_num >999 and frame_num < 10000:
            #      name=dir_path+str(frame_num)+'.png'
            if frame_num<10000:
                
                name=str(frame_num).zfill(5)+'.png'
                name=os.path.join(dir_path,  name)
                print(name)
            cv2.imwrite(name,frame)
            # #ipdb.set_trace()
            prev_frame = frame_copy
            # #ipdb.set_trace()
        cv2.destroyAllWindows()


def YOLOV3(model, frame):
    # #ipdb.set_trace()
    blob = cv2.dnn.blobFromImage(frame, 1/255,(416,416),(0,0,0),swapRB = True,crop= False) # 1,3,416,416 - image not cropped but resized
    # crops image from centre 
    # normalize by dividing by 255
    # mean (0,0,0) substraction and swaps red and blue channels 

    model.setInput(blob) # sets input

    hight,width,_=frame.shape # 1536,2048

    output_layers_name = model.getUnconnectedOutLayersNames() # ('yolo_82', 'yolo_94', 'yolo_106')

    layerOutputs = model.forward(output_layers_name) # input ka answer -  3 ka list 
    # 1. 507,6

    boxes= []
    confidences= []
    class_ids= []

    for output in layerOutputs:
        for detection in output:
            score= detection[5:] 
            class_id= np.argmax(score) # 0
            confidence= score[class_id] 

            if confidence>0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                # #ipdb.set_trace()
                w = int(detection[2] * width)
                h = int(detection[3]* hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.3,.4)
    return [boxes[i]for i in indexes],[confidences[i]for i in indexes], [class_ids[i] for i in indexes]
