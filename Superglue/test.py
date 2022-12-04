# import torch

# a=torch.load('global_registration_sg.pth')
# # a=torch.load('./global_registrationcscsdgergaeda_sg.pth')
# import ipdb;
# ipdb.set_trace()

import cv2
import numpy as np 

img1 = cv2.imread("/home/saharsh2/VLR-Project/Superglue/frame3.jpg")
img2 = cv2.imread("/home/saharsh2/VLR-Project/Superglue/frame4.jpg")

main_box=np.array([       1655,      366.66,      1.2285,      71.898])

detec=np.array([[     1713.5,       720.5,      1.2133,          75],
       [        737,        1163,       0.925,          80],
       [     1331.5,       402.5,     0.97531,          81],
       [        610,         545,      1.0417,          96],
       [      952.5,      1117.5,     0.84615,          91],
       [     1215.5,       261.5,      1.0741,          81],
       [        859,         555,        1.15,          80],
       [     1743.5,        1276,      1.0897,          78]])
for i in detec:
    cv2.rectangle(img2,(int(i[0]),int(i[1])),(int(i[0]+i[2]*i[3]),int(i[1]+i[3])),(255,255,0),2)
# cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
cv2.rectangle(img2,(int(main_box[0]),int(main_box[1])),(int(main_box[0]+main_box[2]*main_box[3]),int(main_box[1]+main_box[3])),(255,0,0),4)
cv2.imwrite('Image.png',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()