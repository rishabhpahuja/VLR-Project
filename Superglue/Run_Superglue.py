import argparse
import numpy as np
import numpy.random as random
import cv2
from super_matching import SuperMatching
import matplotlib.pyplot as plt
import imutils
import pandas as pd
import seaborn as sns
import ipdb
import sys
from deep_sort import linear_assignment


from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
import cv2

# from IPython import embed
# global tt
# tt = 0

# def transf_matrix(theta=0, translation=[0,0]):
#     assert len(translation) == 2
#     tx, ty  = translation

#     # First two columns correspond to the rotation b/t images
#     M = np.zeros((2,3))
#     M[:,0:2] = np.array([[np.cos(theta), np.sin(theta)],\
#                          [ -np.sin(theta), np.cos(theta)]])

#     # Last column corresponds to the translation b/t images
#     M[0,2] = tx
#     M[1,2] = ty
    
#     return M

# """ Convert the 2x3 rot/trans matrices to a 3x3 matrix """
# def transf_mat_3x3(M):
#     M_out = np.eye(3)
#     M_out[0:2,0:3] = M
#     return M_out

# def transf_pntcld(M_est, pt_cld):
#     '''
#     M_est 2x3
#     pt_cld nx2
#     '''
#     R = M_est[:,0:2]
#     t = M_est[:,-1].reshape(2,-1)
#     pt_cld_transf = (R@pt_cld.T + t).T 

#     return pt_cld_transf

# # done
def setup_sg_class(superglue_weights_path):
    # ipdb.set_trace() # being called evry frame
    sg_matching = SuperMatching() # super_matching.py
    sg_matching.weights = 'custom'
    sg_matching.weights_path = superglue_weights_path
    sg_matching.set_weights() # super_matching.py
    
    return sg_matching

def find_conf_values(rect,matches,conf_score,pos,conf):

    k=0
    x0,y0,x1,y1=rect.loc[0],rect.loc[1],rect.loc[2],rect.loc[3]
    for i in range(len(matches)):
        x,y=matches[i]

        if(x>=x0 and x<=x1):
            if(y>y0 and y<=y1):
                k=k+1
                conf_score[pos]+=conf[i]
    if k!=0:
        conf_score[pos]=round(conf_score[pos]/k,3)
    return conf_score

def crop_image(img,box):
    tl,bl=box
    cropped_image=img[tl[1]-5:bl[3]+5,tl[0]-5:bl[2]+5]

    return cropped_image

def increase_rectangle(bbox, pixels,min_=False, max_=False,img_size=(1536,2048)):
    
    x,y=bbox[0],bbox[1]
    if max_:
        x_n,y_n=max(x-pixels,0),max(y-pixels,0)
        return np.array([x_n,y_n])
    
    if min_:
        # import ipdb; ipdb.set_trace()
        x_n,y_n=min(x+pixels,img_size[1]),min(y+pixels,img_size[0])
        return np.array([x_n,y_n])


def Sg_conf(bbox, candidates,frame_t, frame_t_1,sg_matching,reference):

    mconf_row=np.zeros((1,len(candidates)))
    pixels = 50
    bbox_tl, bbox_br = increase_rectangle(bbox[:2],pixels=pixels,max_=True) , increase_rectangle(bbox[:2] + bbox[2:],pixels=pixels,min_=True,img_size=frame_t.shape)
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]
    
    for i in range(len(candidates)):
            # xmin=
            # mconf, _, _, _, _ = sg_matching.detectAndMatch(crop_image(img2_gray,(candidates_tl[i],candidates_br[i])),crop_image(img1_gray,(bbox_tl[i],bbox_br[i])))
            # import ipdb; ipdb.set_trace()
            mconf, _, _, _, _ = SuperGlueDetection(frame_t,frame_t_1,sg_matching,(bbox_tl,bbox_br),\
                                                    (increase_rectangle(candidates_tl[i],pixels=pixels,max_=True),\
                                                    increase_rectangle(candidates_br[i],pixels=pixels,min_=True)),reference=reference)

            # if no detection - mconf is empy- gives back empty and thus, nan is returned - check
            # print(mconf)
            # ipdb.set_trace()
            if (len(mconf)==0):
                 mconf_row[0][i]=0 # if no matches - confidence is zero ??
            else:
                # mconf_row[i]=mconf.cpu().numpy().mean() # changed rish
                mconf_row[0][i]=mconf.mean()#.cpu().numpy().mean() 

    # mconf_row - 
    # array([[    0.61988,     0.57531,     0.55056,      0.5685,       0.571,     0.57331,     0.64581,     0.58366,     0.60006]])
    # print(mconf_row) # nan somehow
    # ipdb.set_trace()
    return mconf_row

def Superglue_cost(tracks, detections, frame_t,frame_t_1 ,track_indices=None,
             detection_indices=None, superglue_weights_path=None,reference=True):
    # ipdb.set_trace()
    if superglue_weights_path is None:
        # raise("SuperGlue Weights Path not given")
        superglue_weights_path = "./global_registration_sg.pth" # path not taken for args when integrating
    # ipdb.set_trace() # calling every frame
    
    sg_matching=setup_sg_class(superglue_weights_path)
    # ipdb.set_trace()

    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - Sg_conf(bbox, candidates, frame_t,frame_t_1,sg_matching,reference=reference)
    return cost_matrix

def SuperGlueDetection(img1, img2, sg_matching,rect1=None ,rect2=None,debug=False,reference=True, showKeyPoints = False):
    # ipdb.set_trace()
    img1_gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) # (1536, 2048)
    img2_gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) # (1536, 2048)

    bbox_tl,bbox_br=rect1
    candidate_tl,candidate_br=rect2
    # ipdb.set_trace()
    #If bounding boxes are passed, a mask is made such that only the fruits are visible to use superglue
    if rect1 is not None:
        image_mask1=np.zeros(img1_gray.shape,np.uint8) # (1536, 2048)
        for i in range(len(rect1)):  # range was blank in rishabh
            image_mask1[int(bbox_tl[1]):int(bbox_br[1]),int(bbox_tl[0]):int(bbox_br[0])]=255
        img1_gray_masked=cv2.bitwise_and(img1_gray,image_mask1) #This mask is for frame_t
        # ipdb.set_trace()

        cv2.imwrite("mask1.png", img1_gray_masked)

        image_mask2=np.zeros(img1_gray.shape,np.uint8)
        # import ipdb;ipdb.set_trace()
        for i in range(len(rect2)):
            image_mask2[int(candidate_tl[1]):int(candidate_br[1]),int(candidate_tl[0]):int(candidate_br[0])]=255
        img2_gray_masked=cv2.bitwise_and(img2_gray,image_mask2) #This mask is for frame_t_11

        cv2.imwrite("mask2.png", img2_gray_masked)

    # This condi
    if rect1 is not None: 
        mconf, kp1, kp2, matches1, matches2 = sg_matching.detectAndMatch(img1_gray_masked, img2_gray_masked,img1_gray,img2_gray,reference)
    
    else:
        mconf, kp1, kp2, matches1, matches2 = sg_matching.detectAndMatch(img1_gray,img2_gray) # only those that match are returned
    
    #! Show matched keypoints
    if (showKeyPoints):
        for ijk,(x,y) in enumerate(kp1.astype(np.int64)):
            cv2.circle(img1, (x,y), 5, (255,0,0), -1)
        for ijk,(x,y) in enumerate(kp2.astype(np.int64)):
            cv2.circle(img2, (x,y), 5, (0,0,255), -1)
        # print(ijk)
        cv2.imwrite("kp1.png", img1)
        cv2.imwrite("kp2.png", img2)

    # ipdb.set_trace()
    # for x,y in kp2.astype(np.int64): 
    # import ipdb; ipdb.set_trace()

    # import ipdb; ipdb.set_trace()
    
    # Show matches
    colours=dict()
    conf_score=[0 for i in range(len(matches1))]
    if debug:
        colour=np.array((sns.color_palette(None,len(rect1))))
        colour=list(np.asarray(colour*255,'uint8'))

        for i in range(len(rect1)):

            conf_score=find_conf_values(rect1.iloc[i], matches1,conf_score,i,mconf)
        
        for j,i in enumerate(range(len(rect1))):

            cv2.rectangle(img1,(rect1.loc[i,0],rect1.loc[i,1]),(rect1.loc[i,2],rect1.loc[i,3]),(int(colour[j][0]),int(colour[j][1]),int(colour[j][2])),3)        
            cv2.putText(img1,str(conf_score[j]),org=(rect1.loc[i,0],rect1.loc[i,1]-2),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(int(colour[j][0]),int(colour[j][1]),int(colour[j][2])),thickness=2)
            colours[j]=((rect1.loc[i,0],rect1.loc[i,1],rect1.loc[i,2],rect1.loc[i,3]),list(colour[j]))
        # import ipdb; ipdb.set_trace()
        sg_matching.plot_matches(img1, img2, kp1, kp2, matches1, matches2,rect1, colours,mconf)

    # ipdb.set_trace()
    
    return mconf,kp1, kp2, matches1, matches2


# """
# Overlay the transformed noisy image with the original and estimate the affine
# transformation between the two
# # """
# # def generateComposite(ref_keypoints, align_keypoints, ref_cloud, align_cloud,
# #                       matches, rows, cols):
# def get_warp_results(ref,align,M_est, debug=False):
#     # Converts vector of keypoints to vector of points or the reverse, where each keypoint is assigned the same size and the same orientation.
    
#     rows, cols = ref.shape

#     M_est_inv = np.linalg.inv(transf_mat_3x3(M_est))[0:2,:]
#     # from IPython import embed; embed()
#     align_warped = cv2.warpAffine(align, M_est_inv, (cols, rows))

#     alpha_img = np.copy(ref)
#     alpha = 0.5
#     composed_img = cv2.addWeighted(alpha_img, alpha, align_warped, 1-alpha, 0.0)
#     if debug:
#         displayImages(composed_img, 'Composite Image')
#     return ref, align_warped, composed_img 

# """
# Compute the translation/rotation pixel error between the estimated RANSAC
# transformation and the true transformation done on the image.
# """
# def computeError(M, M_est, M_est_inv):
#     print('\nEstimated M\n', M_est)
#     print('\nTrue M\n', M)

#     # Add error
#     error = M @ transf_mat_3x3(M_est_inv)
#     R_del = error[0:2,0:2]
#     t_del = error[0:2,2]

#     print('\nTranslation Pixel Error: ', np.linalg.norm(t_del))
#     print('Rotation Pixel Error: ', np.linalg.norm(R_del))
    
# """
# Display a single image or display two images conatenated together for comparison
# Specifying a path will save whichever image is displayed (the single or the
# composite).
# """
# def displayImages(img1, name1='Image 1', img2=None, name2='Image2', path=None):
#     if img2 is None:
#         # ASSERT: Display only 1 image
#         output = img1
#         cv2.namedWindow(name1, cv2.WINDOW_NORMAL)
#         cv2.imshow(name1, img1)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         # Display both images concatenated
#         output = np.concatenate((img1, img2), axis=1)
#         cv2.namedWindow(name1 + ' and ' + name2, cv2.WINDOW_NORMAL)
#         cv2.imshow(name1 + ' and ' + name2, output)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     if path is None:
#         # Save the image at the current path
#         print("")
#     else:
#         cv2.imwrite(path, output)

# """
# Test feature detection, feature matching, and pose estimation on an image.
# """
# def cv_kp_to_np(cv_keypoints):
#     list_kp_np = []
#     for idx in range(0, len(cv_keypoints)):
#         list_kp_np.append(cv_keypoints[idx].pt)
    
#     return np.array(list_kp_np).astype(np.int64)        
#     # ref_cloud = np.float([cv_keypoints[idx].pt for idx in range(0, len(cv_keypoints))]).reshape(-1, 1, 2)


# def find_transformation_SuperGlue(ref, align, sg_matching, debug=False):

#     ref_keypoints, align_keypoints, matches1, matches2 = SuperGlueDetection(ref, align, sg_matching, debug)
    
    
#     try :
#         M_est = cv2.estimateAffinePartial2D(matches1, matches2)[0]
        
  
#     except:
#         print("could not find matches")
#         M_est = np.array([[1,0,0],
#                           [0,1,0]])

#     return M_est, ref_keypoints, align_keypoints, matches1, matches2
 

# def put_at_center(fg_img, bg_img):
#     h, w = fg_img.shape
#     hh, ww = bg_img.shape

#     yoff = round((hh-h)/2)
#     xoff = round((ww-w)/2)

#     result = bg_img.copy()
#     result[yoff:yoff+h, xoff:xoff+w] = fg_img

#     return result


# def resize_images(ref,align):
#     h1,w1 = ref.shape[:2]
#     h2,w2 = align.shape[:2]

#     h = max([h1,w1,h2,w2])

#     bg_img = np.zeros((h,h), dtype=ref.dtype)
#     ref_padded = put_at_center(ref, bg_img)
#     align_padded = put_at_center(align, bg_img)

#     return ref_padded, align_padded


# def adjust_contrast(img):


# # converting to LAB color space
#     lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l_channel, a, b = cv2.split(lab)

#     # Applying CLAHE to L-channel
#     # feel free to try different values for the limit and grid size:
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     cl = clahe.apply(l_channel)

#     # merge the CLAHE enhanced L-channel with the a and b channel
#     limg = cv2.merge((cl,a,b))

#     # Converting image from LAB Color model to BGR color spcae
#     enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

#     # Stacking the original image with the enhanced image
#     # result = np.hstack((img, enhanced_img))
#     # cv2.imshow('test',result)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     return enhanced_img

# def test_two(args):
#     # from IPython import embed; embed()
#     # Load images
#     # ipdb.set_trace()
#     ref_path = args.reference_path ### refernce image - jpeg
#     align_path = args.align_path ### aligning image 

#     ref = cv2.imread(ref_path, 1)  # shape - (1536, 2048, 3)
#     # ref=adjust_contrast(ref) 
#     align = cv2.imread(align_path, 1)
#     # align=adjust_contrast(align)
#     rect1=pd.read_csv('/home/saharsh2/VLR-Project/Superglue/test_data/L0085.csv', header=None) # no. of detections = 36 ... bbox - 4 - (no. of detction boxes, bbox)
#     rect2=pd.read_csv('/home/saharsh2/VLR-Project/Superglue/test_data/L0085.csv', header=None)

#     sg_matching = setup_sg_class(args.superglue_weights_path)
#     ref_keypoints, align_keypoints, matches1, matches2  = SuperGlueDetection(ref, align, sg_matching,rect1,rect2,debug=True)

#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
class SuperGlueClass:
    def __init__(self,superglue_weights_path=None):
        if superglue_weights_path is None:
            # raise("SuperGlue Weights Path not given")
            superglue_weights_path = "./global_registration_sg.pth" # path not taken for args when integrating
        # ipdb.set_trace() # calling every frame
        
        self.sg_matching=setup_sg_class(superglue_weights_path)


    def Superglue_cost(self,tracks, detections, frame_t,frame_t_1 ,track_indices=None,
                detection_indices=None, reference=True):
        # ipdb.set_trace()
        
        # ipdb.set_trace()
        # print("Calculating Superglue Cost")

        if track_indices is None:
            track_indices = np.arange(len(tracks))
        if detection_indices is None:
            detection_indices = np.arange(len(detections))

        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        for row, track_idx in enumerate(track_indices):
            # if tracks[track_idx].time_since_update > 1:
            #     cost_matrix[row, :] = linear_assignment.INFTY_COST
            #     continue

            bbox = tracks[track_idx].to_tlwh()
            candidates = np.asarray([detections[i].tlwh for i in detection_indices])
            cost_matrix[row, :] = 1. - Sg_conf(bbox, candidates, frame_t,frame_t_1,self.sg_matching,reference=reference)
        return cost_matrix
  


# Main Code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image pair pose estimation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # angle=[5,10,15]
    # for i in angle:
    #     img1=cv2.imread("2.png")
    #     img2=imutils.rotate_bound(img1,i)

    parser.add_argument('-ref', '--reference_path',
                        type=str, default='/home/saharsh2/VLR-Project/Superglue/test_data/L0085.jpeg',
                        help='Reference Image')
    parser.add_argument('-align', '--align_path',
                        type=str, default='/home/saharsh2/VLR-Project/Superglue/test_data/L0086.jpeg',
                        help='Image to align')
    # parser.add_argument('--superglue', choices={'indoor', 'outdoor', 'custom'}, 
    #                     default='custom',
    #                     help='SuperGlue weights')

    # parser.add_argument('-weights', '--superglue_weights_path', default='./models/weights/superglue_indoor.pth',
    #                     help='SuperGlue weights path')
    
    parser.add_argument('-weights', '--superglue_weights_path', default='./global_registration_sg.pth',
                        help='SuperGlue weights path')

    
    args = parser.parse_args()
    # test_single(args.reference_path)
    # test_two(args.reference_path, args.align_path)
    # test_two_iterative(args.reference_path, args.align_path)

    # test_two(args)

    video_path='/home/saharsh2/VLR-Project/Superglue/test_video/apples.mp4'

    net=cv2.dnn.readNetFromDarknet("/home/saharsh2/VLR-Project/Superglue/yolo_weights/yolov3.cfg","/home/saharsh2/VLR-Project/Superglue/yolo_weights/yolov3_last.weights")
    tracker=YOLOv7_DeepSORT(reID_model_path="/home/saharsh2/VLR-Project/Superglue/deep_sort/model_weights/mars-small128.pb", detector=net)

    tracker.track_video(video_path, output="./IO_data/output/SG_C0_48_expand_10_rect_with_det.avi",show_live =False, skip_frames = 0, count_objects = True, verbose=1,dir_path='/home/saharsh2/VLR-Project/Superglue/deep_sort/Tests/')

    
    