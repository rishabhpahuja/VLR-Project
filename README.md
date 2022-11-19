# VLR-Project
This repository is used for the project "SuperDeepSort" for Visual Learning and Recognition (CMU-16824) (Class Link: https://visual-learning.cs.cmu.edu/hw1.html)

## Superglue ##
This folder consists of the script to use superglue algorithm. To run this script use the following command:

`python3 Run_superglue.py`

This script shall require two pre-trained weights, one of which is already present at the desired location. The second set of weights can be downloaded from the link:
(https://drive.google.com/drive/folders/1WzrzGSEhuEaT8cDBWh1PfyI9nx9kbw1J?usp=share_link). The location of these weights can be passed as an argument while runnng the file Run_Superglue.py 

### **Arguments:** ###
1. **ref:** The reference image wrt to which the features of another image have to be matched to
2. **align:** The image whose features have to be matched to reference image
3. **weights:** The location of superglue weights
