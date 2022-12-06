import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from tqdm import tqdm

def get_images(video_path):

    images=[]

    try: # begin video capture
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)
    

    while True: # while video is running
        
        return_value, frame = vid.read()
        images.append(frame)

        if not return_value:
            print('Video has ended !')
            break
    return images
    


if __name__=='__main__':
    video_1='/home/swathi/CMU/vlr/vlr-project/VLR-Project/VLR-Project/Superglue/test_video/Stepladder_work.MP4'
    video_2='/home/swathi/CMU/vlr/vlr-project/VLR-Project/VLR-Project/Superglue/test_video/Stepladder_work.MP4'
    vid1_images=get_images(video_1)
    vid2_images=get_images(video_2)
    ipdb.set_trace()
    assert len(vid1_images)==len(vid2_images)
    height, width, layers = vid1_images[0].shape
    size = (width*2,height)
    out = cv2.VideoWriter('Merged_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
 
    for i in tqdm(range(len(vid1_images))):
        out.write(np.concatenate((vid1_images[i],vid2_images[i]),axis=1))
    out.release()






