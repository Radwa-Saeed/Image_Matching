import time
from  SIFT import *
from Utilities.Read_Show import Read_Img
import cv2
import matplotlib.pyplot as plt

img = Read_Img("./images/Sift2.jpg")

t1 = time.time()
keypoints, descriptors = SIFT.generateFeatures(img)         #our SIFT 
t2 = time.time()

print("Execution time of SIFT is {} sec".format(t2 - t1))


rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # because openCV use BGR but matplot use RGB {this is only for showing the image}
imgplot = plt.imshow(rgbImg)

for pnt in keypoints:
    plt.scatter(pnt.pt[0], pnt.pt[1], s=pnt.size, c="red")

plt.show()
