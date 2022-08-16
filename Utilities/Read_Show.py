import cv2
from matplotlib import pyplot as plt
import numpy as np


def Read_Img(path):
    """
    Read The Image And Return It Grey-Scalled
    """
    return cv2.imread(path,cv2.IMREAD_COLOR)
    # return cv2.imread(path)

def Show_Img(Window_Name:str,Image:np.ndarray):
    """
    Display only one window at time 
    """
    cv2.namedWindow(Window_Name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Window_Name, 500, 500)
    cv2.imshow(Window_Name, Image)
    

def Display_Img(Title,img):
    plt.figure(figsize = (10,10))
    img = plt.imshow(img,cmap='gray', vmin = 0, vmax = 255,interpolation='none')
    plt.axis('off')
    plt.title(Title)
    return img
    

def save_image(image_save_path, image):         #function takes the image and the path in which it will be saved and save the image in this path
        cv2.imwrite(image_save_path, image)     #the path should include the name like ( /images/"image.png" )

def Histogram_plot(Num_Bars,Hights):
    plt.bar(Num_Bars, Hights, width = 1, color = ['grey', 'black'])
    plt.xlabel('Pixels')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()

def Distribution_plot (X,Y):
    plt.plot(X, Y)
    plt.xlabel('Intensities')
    plt.ylabel('Frequency')
    plt.title('Distribution curve')
    plt.show()