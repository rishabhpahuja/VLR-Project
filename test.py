import cv2
import numpy as np
from skimage import exposure

def match_hist(ref_img_path, img_path, display=True, save=True):

    ref_img= cv2.imread(ref_img_path,1)
    img=cv2.imread(img_path,1)
    multi= True if ref_img.shape[-1]>1 else False
    matched = exposure.match_histograms(img, ref_img, multichannel=multi)

    if display:
        cv2.namedWindow("Output",cv2.WINDOW_NORMAL)
        cv2.imshow('Output',matched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if save:
        cv2.imwrite(img_path,matched)

for i in ['L0067.jpeg','L0068.jpeg','L0069.jpeg','L0070.jpeg']:
    # img1=match_hist('_undistorted.jpg',i)
    img1=cv2.imread(i,1)
    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
    cv2.imshow('Image',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()