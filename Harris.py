from Utilities.Read_Show import Read_Img
from Utilities.convolve2D import convolve2D
from Utilities.filters import gaussian_filter, sobel_filter
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time


def Harris_Edge_Detector(Img_Path,Window_Size=3,K=0.5):
    harris_time_start = time.time()
    """ Compute Harris operator using hessian matrix of the image
    input : image
    Return: Harris operator
    """
    #Img= Read_Img(Img_Path)
    src = cv2.imread(Img_Path)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    print('Image successfully read...')
    
    # 1.calculate Ix , Iy ( Dervatives in X & Y direction)...Sobel
    # Ix = sobel_filter(img,kernel_size=3,direction="x")
    # Iy = sobel_filter(img,kernel_size=3,direction="y")
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Window_Size)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Window_Size)

    # 2. Hessian Matrix Calculation H=([Ixx , Ixy],[Ixy , Iyy]) where Ixx = Ix^2 ...
    Ixx=np.multiply(Ix,Ix)
    Iyy=np.multiply(Iy,Iy)
    Ixy=np.multiply(Ix,Iy)

    # 3. Image Smoothing (Gaussian Filter)
    # Ixx = gaussian_filter(Ixx,kernel_size=Window_Size)
    # Iyy = gaussian_filter(Iyy,kernel_size=Window_Size)
    # Ixy = gaussian_filter(Ixy,kernel_size=Window_Size)
    Ixx = cv2.GaussianBlur(Ixx,(Window_Size,Window_Size),0)
    Iyy = cv2.GaussianBlur(Iyy,(Window_Size,Window_Size),0)
    Ixy = cv2.GaussianBlur(Ixy,(Window_Size,Window_Size),0)

    print ("Finding Corners...")
    # 4. Computing Response Function [ R = det(H) - k*(Trace(H))^2 ]
    det_H = Ixx*Iyy - Ixy**2
    trace_H = Ixx + Iyy
    R = det_H - K*(trace_H**2) 
    harris_time_end = time.time()
    print(f"Execution time of Harris Algorithm is {harris_time_end - harris_time_start}  sec")
    return R

def Img_Features(Img_Path,Response_Mat,Threshold=2.5):
    src = cv2.imread(Img_Path)
    # 5. Select large values of R [ relative to maximum value : corners = np.abs(R) > 0.2 * np.max(R)]
    # R_Mat = cv2.dilate(R, None)
    max_corner = np.max(Response_Mat)
    corner = np.array(Response_Mat > (max_corner * Threshold), dtype="int8")
    # edges = np.array(R < 0, dtype="int8")
    # flat= np.array(R == 0, dtype="int8")
    # 6. Mark each corner with colored pixel [255,0,0] on the src image
    src = src[:corner.shape[0], :corner.shape[1]]
    src[corner == 1] = [255,0,0]

    # 7. Plt the image with corner pixels
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(src)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, src.shape[1])
    ax.set_ylim(src.shape[0], 0)
    plt.show()

    
    # return "Done"
# path1 = "img/chess2.jpeg"
# R_Mat= Harris_Edge_Detector(path1,3,0.05)
# print(R_Mat)
# print(Harris_Edge_Detector("img/shapes.jpg",5,0.05,0.01))
# print(Harris_Edge_Detector("img/chess.png",5,0.05,0.01))
# print(Harris_Edge_Detector("img/leaf.jpeg",5,0.05,0.01))

# img = "img/chess2.jpeg"
# R_Mat= Harris_Edge_Detector(img,3,0.05)
# Img_Features(img,R_Mat,0.01)
# print(R_Mat)

# plt.figure("Original Image")
# plt.imshow(img)
# plt.figure("Ixx")
# plt.imshow(Ixx)
# plt.set_cmap("gray")
# plt.figure("Iyy")
# plt.imshow(Iyy)
# plt.figure("Ixy")
# plt.imshow(Ixy)
# plt.figure("Harris Operator")
# plt.imshow(np.abs(R))
# plt.show()