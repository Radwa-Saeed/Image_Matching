from cv2 import pencilSketch, sqrt
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from scipy import signal
width = 0
height=0
h=0
w=0


# original_img = cv2.imread("./images/cat256.jpg")
# template_img =cv2.imread("./images/cat256_edited_v1.png")

original_img = cv2.imread("./images/Sift.jpg")
template_img =cv2.imread("./images/Sift_ori.jpg")

original_img = cv2.resize(original_img,(256,256))
template_img = cv2.resize(template_img,(256,256))

original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create() 
kp1, descriptor1 = sift.detectAndCompute(original_img, None)
kp2, descriptor2 = sift.detectAndCompute(template_img, None)
# print("1",des1)
# print("2",des2)
# print("1",len(des1))
# print("2",len(des2))


def calculateSSD(desc_image1,desc_image2):
    sum_square = 0
    for m in range(len(desc_image2)-1):
        sum_square += (desc_image1[m] - desc_image2[m]) ** 2
        # difference = desc_image1[m] - desc_image2[m]
        # sum_square = np.sum(np.square(difference))
    # The (-) sign here because the condition we applied after this function call is reversed
    SSD = - (np.sqrt(sum_square))
    return SSD
    

SSD = calculateSSD(descriptor1,descriptor2)
# print("S",SSD)
# print(len(SSD))

def calculate_NCC(desc_image1, desc_image2):


    normlized_output1 = (desc_image1 - np.mean(desc_image1)) / (np.std(desc_image1))
    normlized_output2 = (desc_image2 - np.mean(desc_image2)) / (np.std(desc_image2))
    correlation_vector = np.multiply(normlized_output1, normlized_output2)
    NCC = float(np.mean(correlation_vector))

    return NCC



def feature_matching_temp (descriptor1,descriptor2,method):


    keyPoints1 = descriptor1.shape[0]
    keyPoints2 = descriptor2.shape[0]

    #Store matching scores
    matched_features = []

    for kp1 in range(keyPoints1):
        # Initial variables (will be updated)
        distance = -np.inf
        y_index = -1
        for kp2 in range(keyPoints2):
            # Choose methode (ssd or normalized correlation)
            if method=="SSD":
               score = calculateSSD(descriptor1[kp1], descriptor2[kp2])
            elif method =="NCC":
                score = calculate_NCC(descriptor1[kp1], descriptor2[kp2])


            if score > distance:
                distance = score
                y_index = kp2

        feature = cv2.DMatch()
        #The index of the feature in the first image
        feature.queryIdx = kp1
        # The index of the feature in the second image
        feature.trainIdx = y_index
        #The distance between the two features
        feature.distance = distance
        matched_features.append(feature)

    return matched_features

# Apply feature matching using SSD ::
# matched_features = feature_matching_temp(descriptor1, descriptor2,"SSD")
# matched_features = sorted(matched_features, key=lambda x: x.distance, reverse=True)
# matched_image = cv2.drawMatches(original_img, kp1, template_img, kp2,matched_features[:30], template_img, flags=2)

# plt.imshow(matched_image)
# plt.show()

# # Apply feature matching using NCC ::
# matched_features = feature_matching_temp(descriptor1, descriptor2,"NCC")
# matched_features = sorted(matched_features, key=lambda x: x.distance, reverse=True)
# matched_image = cv2.drawMatches(original_img, kp1, template_img, kp2,matched_features[:30], template_img, flags=2)

# plt.imshow(matched_image)
# plt.show()



