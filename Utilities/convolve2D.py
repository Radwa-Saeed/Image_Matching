import numpy as np

def convolve2D (image,kernel,padding = True ):
    image_h = image.shape[0]    #getting image height
    image_w = image.shape[1]    #getting image width

    kernel_h = kernel.shape[0]  #getting kernel height
    kernel_w = kernel.shape[1]  #getting kernel width

    start_h = kernel_h // 2     #this is the row from which iterration will start
    start_w = kernel_w // 2     #this is the col from which iterration will start

    convolved_image = np.zeros(image.shape)  #initialize the convolved image with zeroes and that will result in zero padding

    for row_index in range(start_h, image_h - start_h):         #START FROM SECOND ROW if kernel is 3x3
        for col_index in range (start_w, image_w - start_w):    #START FROM SECOND COL if kernel is 3x3    (this means that we will put the middle value of kernel on image[1][1])

            sum = 0     #which will carry the matrix dot product result between kernel and pixels under it

            for i in range(kernel_h):       #iterate over the kernel rows
                for j in range(kernel_w):   #iterate over the kernel cols
                    sum = sum + kernel[i][j] * image [row_index - start_h + i][col_index - start_w + j]         #multiply each value in kernel by the image value under it and sum all 

            convolved_image[row_index][col_index] = sum     #put the result value as the new pixel value
    
    if padding == False:
        convolved_image = convolved_image[start_h:-start_h, start_w:-start_w]       #remove the zero padding

    else: 
        pass

    return convolved_image
