# VLR-Project
This repository is used for the project "SuperDeepSort" for Visual Learning and Recognition (CMU-16824) (Class Link: https://visual-learning.cs.cmu.edu/hw1.html)

**A.** In this project we are trying to incorporate superglue in deep sort for re-association of objects. We are trying different methods of incorporating superglue algorithm in the baseline. We tried the following things:

1. Replace cosine distance metric to find cost matrix by superglue algorithm
2. Replace IOU metric to find cost matrix by superglue algorithm 
3. Add superglue and cosine matrix by assigning weights and then send the unmatched boxes to IOU.

### Note:
-The script for **A.1** has been added. TO activate it, change True->False in line 122 of ./deep_sort/deep_sort/tracker.py
-The script has been written to implement **A.2**. To activate this, change False->True in line 145 of ./deep_sort/deep_sort/tracker.py
-The script has been written to implement **A.3**. To activate this, change False->True in line 132 of ./deep_sort/deep_sort/tracker.py

#### Pendign tasks:
- Passing frames to use any part of **A**
- Decide whether to use gated_metric for superglue. It hasn't been added yet.

**B.** These cases can further be tried by two cases:
1. We apply superglue between the two frames by cropping the images along the bounding boxes.
2. We apply superglue between the two frames such that keypoints are found along the cropped images and keypoint matching is applied on the whole frame. 

### Note:
The scripts right now are written such that the keypoints are found within the fruit region and matched using the entire image. To change this to case where only bounding box area is passed to superglue, call Superglue_cost function by passing reference=False. 

## Superglue ##
This folder consists of the script to use superglue algorithm. To run this script use the following command:

`python3 Run_superglue.py`

This script shall require two pre-trained weights, one of which is already present at the desired location. The second set of weights can be downloaded from the link:
(https://drive.google.com/drive/folders/1WzrzGSEhuEaT8cDBWh1PfyI9nx9kbw1J?usp=share_link). The location of these weights can be passed as an argument while runnng the file Run_Superglue.py 

### **Arguments:** ###
1. **ref:** The reference image wrt to which the features of another image have to be matched to
2. **align:** The image whose features have to be matched to reference image
3. **weights:** The location of superglue weights
