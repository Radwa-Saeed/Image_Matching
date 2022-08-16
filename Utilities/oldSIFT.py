# def generate_gauss_octave(img,intervalsNO = 5, s = 1.4, sigma = 1.6):
#     octave = []          #initialize octave with the base image 
#     from cv2 import GaussianBlur

#     for k in range (intervalsNO):       #to generate the right number of ntervals after appling DoG
#         sigma_k = sigma * (s**k)            #computing sigma
#         blurred = GaussianBlur(img, (0, 0), sigmaX = sigma_k, sigmaY = sigma_k)
#         octave.append(blurred)              #appending the image in the octave
#         # cv2.imshow("Scale Space K = {}".format(k),blurred)
#         # cv2.waitKey(0)
#     return octave


# def gauss_pyramid(image, octavesNO = 4, s = 1.4, sigma = 1.6):
#     pyramid = []    #initialize pyramid
#     import numpy as np
#     img = np.copy(image)    #copy the image

#     for _ in range(octavesNO):                  #looping over the pyramid layers
#         octave = generate_gauss_octave(img = img,s = s,sigma = sigma)   #generate octave
#         pyramid.append(octave)                  #append it in the pyramid
#         img = img[::2,::2]                      #downscaling the image by taking even rows and cols

#     return pyramid



# def DoG_pyramid(gauss_pyr):
#     DoG_pyr = []

#     for gauss_octave in gauss_pyr:
#         DoG_octave = []

#         for i in range(len(gauss_octave) - 1):
#             DoG_octave.append(gauss_octave[i+1] - gauss_octave[i])
        
#         DoG_pyr.append(DoG_octave)

#     return DoG_pyr


# def getPixelNeighbours(octave, image_idx, x, y):
#     neighbours = []
#     img = octave[image_idx]

#     neighbours += [
#         img[x+1,y],
#         img[x-1,y],
#         img[x,y+1],
#         img[x,y-1],
#         img[x+1,y+1],
#         img[x-1,y-1],
#         img[x+1,y-1],
#         img[x-1,y+1]
#     ]

#     if image_idx != 0:
#         previouse_img = octave[image_idx - 1]

#         neighbours += [
#         previouse_img[x+1,y],
#         previouse_img[x-1,y],
#         previouse_img[x,y+1],
#         previouse_img[x,y-1],
#         previouse_img[x+1,y+1],
#         previouse_img[x-1,y-1],
#         previouse_img[x+1,y-1],
#         previouse_img[x-1,y+1]
#     ]

#     if image_idx != (len(octave) - 1):
#         next_image = octave[image_idx + 1]

#         neighbours += [
#         next_image[x+1,y],
#         next_image[x-1,y],
#         next_image[x,y+1],
#         next_image[x,y-1],
#         next_image[x+1,y+1],
#         next_image[x-1,y-1],
#         next_image[x+1,y-1],
#         next_image[x-1,y+1]
#     ]

#     return neighbours

# def getOctavesMaxMinPoints(pyr):
#     points = []

#     for octave_idx in range(len(pyr)):
#         print(octave_idx)
#         octave = pyr[octave_idx]
#         for image_idx in range(len(octave)):
#             image = octave[image_idx]
#             for x in range(1, image.shape[0] - 1):
#                 for y in range(1, image.shape[1] - 1):
#                     pixel = image[x,y]
#                     neighbours = getPixelNeighbours(octave,image_idx,x,y)
#                     max = True 
#                     min = True
#                     for n in neighbours:
#                         if n >= pixel:
#                     #if any(n > pixel for n in neighbours):
#                             max = False
#                     #if any(n < pixel for n in neighbours):
#                         if n <= pixel:
#                             min = False
#                     if max or min:
#                         #print([octave_idx,image_idx,x,y]) 
#                         points.append([octave_idx,image_idx,x,y])
        
#     return(points)



# import cv2
# path = "./images/0.png"
# im = cv2.imread(path)
# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# gaussPyr = gauss_pyramid(gray)
# dog = DoG_pyramid(gauss_pyr=gaussPyr)
# maxMin = getOctavesMaxMinPoints(dog)

# print(im.shape)
# print(dog[0][0].shape)
# for point in maxMin:
#     x = point[-2]
#     y = point[-1]
#     image = cv2.circle(im, (y,x), radius=0, color=(0, 0, 255), thickness=-1)

# # print(len(dog[0]))
# cv2.imshow("DoG",image)
# cv2.waitKey(0)

