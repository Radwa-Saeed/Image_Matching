import numpy as np
from Utilities.kernels import *
from Utilities.convolve2D import *



def average_filter(image, kernel_size=3):
    kernel = average_kernel(kernel_size = kernel_size)
    output = convolve2D(image, kernel)

    return output.astype(np.uint8)


def gaussian_filter(image, kernel_size=3, std=1):
    if std == 0:
            raise ValueError("Standard Diviation can't be zero")   
    kernel = gaussian_kernel(kernel_size = kernel_size, std = std)
    output = convolve2D(image, kernel)

    return output.astype(np.uint8)

           
def median_Filter(data, filter_size):
    """""
    Get the median filter by taking filter size as a parameter and return median value after sorting pixels in
    ascending order  
    """
    
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            data_final= data_final.astype(np.uint8)
            temp = []
    return data_final

    output = convolve2D(image, median_kernel)
    return output.astype(np.uint8)

def perwitt_filter(image, kernel_size = 3, direction = "x"):
    if direction.lower() not in ["x", "y", "xy"]:
            raise ValueError("direction must be 'x' or 'y' or 'xy' ")   # direction must be specified

    if direction.lower() == "xy":
        return perwitt_filter(image, kernel_size = kernel_size, direction="x") + perwitt_filter(image, kernel_size = kernel_size, direction="y")
    else:
        kernel = prewitt_kernel(kernel_size = kernel_size, direction = direction)
        output = convolve2D (image, kernel)
   
        #output = np.clip(output,0,255)
        output = np.absolute(output)

        output = ((output-output.min())/(output.max()-output.min())) * 255

    return output.astype(np.uint8)


def sobel_filter(image, kernel_size = 3, direction = "x"):
    if direction.lower() not in ["x", "y", "xy"]:
        raise ValueError("direction must be 'x' or 'y' or 'xy' ")   # direction must be specified

    if direction.lower() == "xy":
        return sobel_filter(image, kernel_size = kernel_size, direction="x") + sobel_filter(image, kernel_size = kernel_size, direction="y")

    else:
        kernel = sobel_kernel(kernel_size = kernel_size, direction = direction)
        output = convolve2D (image, kernel)

        #output = np.clip(output,0,255)
        output = np.absolute(output)

        #output = ((output-output.min())/(output.max()-output.min())) * 255
    return output.astype(np.uint8)


def roberts_filter(image, direction = "x"):
    if direction.lower() not in ["x", "y", "xy"]:
        raise ValueError("direction must be 'x' or 'y' or 'xy' ")   # direction must be specified

    if direction.lower() == "xy":
        return roberts_filter(image, direction="x") + roberts_filter(image, direction="y")

    else:
        kernel = roberts_kernel(direction = direction)
        output = convolve2D (image, kernel)
        
        #output = np.clip(output,0,255)
        output = np.absolute(output)

        output = ((output-output.min())/(output.max()-output.min())) * 255

    return output.astype(np.uint8)

    
def non_max_suppression_filter(image, direction):
       
    Rows, Columns = image.shape[0], image.shape[1]
    Z = np.zeros((Rows, Columns), dtype=np.int32) 

    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for row in range(1, Rows-1):
        for column in range(1, Columns-1):
            try:
                first_neighbor = 255
                second_neighbor = 255

                # angle 0
                if (0 <= angle[row, column] < 22.5) or (157.5 <= angle[row, column] <= 180):
                    first_neighbor = image[row, column+1]
                    second_neighbor = image[row, column-1]
                # angle 45
                elif (22.5 <= angle[row, column] < 67.5):
                    first_neighbor = image[row+1, column-1]
                    second_neighbor = image[row-1, column+1]
                # angle 90
                elif (67.5 <= angle[row, column] < 112.5):
                    first_neighbor = image[row+1, column]
                    second_neighbor = image[row-1, column]
                # angle 135
                elif (112.5 <= angle[row, column] < 157.5):
                    first_neighbor = image[row-1, column-1]
                    second_neighbor = image[row+1, column+1]

                if (image[row, column] >= first_neighbor) and (image[row, column] >= second_neighbor):
                    Z[row, column] = image[row, column]
                else:
                    Z[row, column] = 0

            except IndexError as e:
                pass

    return Z



def canny_threshold(image,low_threshold_ratio=0.05, high_threshold_ratio=0.1):
       
        high_threshold = image.max() * high_threshold_ratio
        low_threshold = high_threshold * low_threshold_ratio

        A, B = image.shape[0], image.shape[1]
        res = np.zeros((A, B), dtype=np.int32)

        weak_pixels = np.int32(25)
        strong_pixels= np.int32(255)

        high_i, high_j = np.where(image >= high_threshold)
        #low_i, low_j = np.where(image < low_threshold)

        weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold)) #those are the weak pixels between the two thresholds

        res[high_i, high_j] = strong_pixels

        res[weak_i, weak_j] = weak_pixels

        return (res, weak_pixels, strong_pixels)


def hysteresis(image, weak = 25, strong = 255):
        image = np.copy(image)
        rows, columns = image.shape[0], image.shape[1]
        for row in range(1, rows-1):
            for col in range(1, columns-1):
                if (image[row, col] == weak):
                    try:
                        if ((image[row+1, col-1] == strong) or (image[row+1, col] == strong) or (image[row+1, col+1] == strong)
                            or (image[row, col-1] == strong) or (image[row, col+1] == strong)
                                or (image[row-1, col-1] == strong) or (image[row-1, col] == strong) or (image[row-1, col+1] == strong)):

                            image[row, col] = strong
                        else:
                            image[row, col] = 0
                    except IndexError as e:
                        pass
        return image


def canny_filter(image,std = 1, low_ratio=0.05, high_ratio=0.1):
    
    if image.any() != None:
        if(len(image.shape)<2):
            print ('grayscale')

        elif len(image.shape)==3:
            import cv2
            print ('Colored')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        print("cannot find image") 

    smoothed = gaussian_filter(image,kernel_size = 5, std = std)

    grad_x = sobel_filter(smoothed, kernel_size = 3, direction = "x")
    grad_y = sobel_filter(smoothed, kernel_size = 3, direction = "y")

    Grad_xy = np.hypot(grad_x, grad_y)
    theta = np.arctan2(grad_y, grad_x)

    suppressed = non_max_suppression_filter(Grad_xy, theta)

    thresholded, weak_pix, strong_pix = canny_threshold(suppressed, low_threshold_ratio = low_ratio, high_threshold_ratio = high_ratio)

    output = hysteresis(thresholded, weak = weak_pix, strong = strong_pix)

    return output.astype(np.uint8)



def global_threshold(image, threshold_value, max = 255):
    #Global threshold function takes image and threshold value and maximum value as parameters and return img after applying threshold

    img = np.copy(image)
    img[img[:,:] < threshold_value] = 0         #set pixels value that are less than threshold to zero
    img[img[:,:] >= threshold_value] = max      #set pixels value that are greater than threshold to maximum

    return img


def local_threshold(image, block_size, max = 255):
    img = np.copy(image)                            #copy of image to work on without changing the original 
    V_blocks = np.ceil(img.shape[0] / block_size )          #number of blocks in Horizontal axis 
    H_blocks = np.ceil(img.shape[1] / block_size )          #number of blocks in Vertical axis
    for i in range(int(V_blocks)):               #loop on Vertical blocks
        for j in range(int(H_blocks)):           #loop on Horizontal blocks
            block = img[(block_size * i) : (block_size * (i + 1)),(block_size * j) : (block_size * (j + 1))]    #fill the block with the pixels values 
            tresh = block.mean()                                                                          #take mean of the values to be threshold
            block[block < tresh] = 0                    #pixels values less than threshold will be set to zero
            block[block >= tresh] = max                 #pixels values more than threshold will be set to max
 
    return img
