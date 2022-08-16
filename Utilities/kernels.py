import numpy as np



def average_kernel(kernel_size=3):

    kernel = np.ones((kernel_size,kernel_size))     #initialize kernel with ones with number of col and rows equal to kernel size
    avg_kernel = kernel/np.sum(kernel)              #divide by sum to get average
    return avg_kernel                               #return avg kernel



def gaussian_kernel(kernel_size=3, std=1):

    k = kernel_size // 2

    x = np.arange(- k, k +1, 1)
    y = np.arange(-kernel_size, kernel_size+1, 1)

    x, y = np.meshgrid(x, y)
    # gauss_kernel = (1/(2*np.pi*std**2))*np.exp(-((x**2+y**2)/(2*std**2)))
    gauss_kernel = np.exp(- ((x**2 + y**2) / (2 * std**2)))
    gauss_kernel = gauss_kernel/np.sum(gauss_kernel)
  
    return gauss_kernel    


def prewitt_kernel(kernel_size=3, direction="x"):  # function that takes perwitt kernel size as parameter (default size = 3) and direction (default x direction)
    if direction.lower() not in ["x", "y"]:  # and return kernal as ndarray
        raise ValueError("Undefined direction, use x or y")  # direction must be either in x or y axis
    a = np.ones((kernel_size, 1))  # creating col of ones
    b = np.arange(- (kernel_size // 2), ((kernel_size // 2) + 1), 1).reshape(1, kernel_size )  # creating a row of perwitt kernel values
    prewitt_kernel = (a @ b if direction.lower() == "x" else (a @ b).T)  # matrix multiplication of a and b to get the kernel matrix

    return prewitt_kernel
    


def sobel_kernel(kernel_size=3, direction="x"):  # function that takes sobel kernel size as parameter (default size = 3) and direction (default x direction)
    if direction.lower() not in ["x", "y"]:  # and return kernal as ndarray
        raise ValueError("Undefined direction, use x or y")     # direction must be either in x or y axis
    a = np.ones((kernel_size, 1))  # creating col of ones
    a[(kernel_size // 2), :] = 2  # putting 2 in the middle so that the middle of the kernel is weighted by 2
    b = np.arange(- (kernel_size // 2), ((kernel_size // 2) + 1), 1).reshape(1, kernel_size)  # creating a row of sobel kernel values
    sobel_kernel = (a @ b if direction.lower() == "x" else (a @ b).T)  # matrix multiplication of a and b to get the kernel matrix

    return sobel_kernel


def roberts_kernel(direction="x"):  # function that takes roberts kernel direction (default x direction) and return kernal as ndarray
    if direction.lower() not in ["x", "y"]:
        raise ValueError("Undefined direction, use x or y")  # direction must be either in x or y axis
    roberts_kernel_x = np.array([[1, 0], [0, -1]])  # roberts kernel in x direction
    roberts_kernel_y = np.array([[0, 1], [-1, 0]])  # roberts kernel in y direction
    roberts_kernel = roberts_kernel_x if direction.lower() == "x" else roberts_kernel_y

    return roberts_kernel

