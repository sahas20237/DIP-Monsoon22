#All codes DIP  Section B Monsoon 22

'''
1. Mean filter (Blurring)

2. Median filter (Denoising)

3. Edge detection

4. Sharpening

5. Blurring: Gaussian, Motion, Out of focus

6. Salt and pepper Noise

7. Interpolation: Nearest Neighbour, Bilinear, Bicubic, Bell, Hermite, Lanczos

8. FFT/IFFT Filters not doing

'''

#Genreal imports
import math
import random
import cv2
import numpy as np
import scipy.signal as signal

def show_image(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def GrayAndPadded(img_, filterLength):
    img = img_.copy()
    l1, b1, c1 = img.shape
    l2 = b2 = filterLength

    #Converting image to Gray to get binary (0-255) matrix values
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Creating the padding matrix of zeros
    padding = np.zeros(((l1+l2-1), (b1+b2-1)))

    #Inserting binary values (0-255) in padding matrix
    for i in range(l1):
        for j in range(b1):
            padding[i+l2//2, j+b2//2] = gray_img[i, j]

    padding = padding.astype(np.uint8)
    return padding

def edge_detection_and_sharpen_FO(img_, filter_x, filter_y):
    img = img_.copy()
    l1, b1 = img.shape
    l2, b2 = filter_x.shape
    final = np.zeros((l1, b1))

    for i in range(l1-l2+1):
        for j in range(b1-b2+1):
            #Creating a box of filter sized from the image matrix
            temp_box1 = img[i:i+l2, j:j+b2]
            temp_box2 = img[i:i+l2, j:j+b2]

            #Multiplying the box with the filter
            filtered_tempbox1 = np.sum(temp_box1*filter_x)
            filtered_tempbox2 = np.sum(temp_box2*filter_y)

            #Assigning the value to the center of the box by taking the arguement
            final[i+l2//2][j+b2//2] = math.sqrt(filtered_tempbox1**2 + filtered_tempbox2**2)

    final = img - final

    final = final.astype(np.uint8)
    return final

def edge_detection_and_sharpen_SO(img_, filter_x):
    img = img_.copy()
    l1, b1 = img.shape
    l2, b2 = filter_x.shape
    final = np.zeros((l1, b1))

    for i in range(l1-l2+1):
        for j in range(b1-b2+1):
            #Creating a box of filter sized from the image matrix
            temp_box = img[i:i+l2, j:j+b2]

            #Multiplying the box with the filter
            filtered_tempbox = np.sum(temp_box*filter_x)

            #Assigning the value to the center of the box by taking the arguement
            final[i+l2//2][j+b2//2] = filtered_tempbox
    
    final = img - final

    final = final.astype(np.uint8)
    return final

def median3x3(img_):
    img = img_.copy()
    l1, b1 = img.shape
    final = np.zeros((l1, b1))

    filter3x3 = np.array([(0, 0, 0),
                          (0, 0, 0),
                          (0, 0, 0)])
    l2, b2 = filter3x3.shape

    for i in range(l1):
        for j in range(b1):
            temp_box = img[i:i+l2, j:j+b2]
            filtered_tempbox = np.median(temp_box)
            final[i, j] = filtered_tempbox

    return final

# Median Filter as used in the previous assignment
def median5x5(img_):
    img = img_.copy()
    l1, b1 = img.shape
    final = np.zeros((l1, b1))

    filter5x5 = np.array([(0, 0, 0, 0, 0), 
                          (0, 0, 0, 0, 0),
                          (0, 0, 0, 0, 0),
                          (0, 0, 0, 0, 0), 
                          (0, 0, 0, 0, 0)])
    l2, b2 = filter5x5.shape

    for i in range(l1):
        for j in range(b1):
            temp_box = img[i:i+l2, j:j+b2]
            filtered_tempbox = np.median(temp_box)
            final[i, j] = filtered_tempbox

    return final

def mean3x3(img_):
    img = img_.copy()
    l1, b1 = img.shape
    final = np.zeros((l1, b1))

    filter3x3 = np.array([(1, 1, 1), 
                          (1, 1, 1), 
                          (1, 1, 1)])*(1/9)
    l2, b2 = filter3x3.shape

    for i in range(l1):
        for j in range(b1):
            temp_box = img[i:i+l2, j:j+b2]
            filtered_tempbox = np.sum(temp_box*filter3x3)
            final[i, j] = filtered_tempbox

    return final

def mean5x5(img_):
    img = img_.copy()
    l1, b1 = img.shape
    final = np.zeros((l1, b1))

    filter5x5 = np.array([(1, 1, 1, 1, 1), 
                          (1, 1, 1, 1, 1),
                          (1, 1, 1, 1, 1),
                          (1, 1, 1, 1, 1), 
                          (1, 1, 1, 1, 1)])*(1/25)
    l2, b2 = filter5x5.shape

    for i in range(l1):
        for j in range(b1):
            temp_box = img[i:i+l2, j:j+b2]
            filtered_tempbox = np.sum(temp_box*filter5x5)
            final[i, j] = filtered_tempbox

    return final

def motio_blur(img_):
    img = img.copy()
    l1, b1 = img.shape
    final = np.zeros((l1, b1))

    kernel = np.array([ (0, 0, 0, 0, 0, 0, 0, 0, 0, 1), 
                        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
                        (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
                        (0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
                        (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
                        (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
                        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
                        (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
                        (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
                        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)])*(1/10)
    l2, b2 = kernel.shape

    for i in range(l1):
        for j in range(b1):
            temp_box = img[i:i+l2, j:j+b2]
            filtered_tempbox = np.sum(temp_box*kernel)
            final[i, j] = filtered_tempbox

    return final

def out_of_focus(img_, radius, C):
    img = img_.copy()
    l1, b1 = img.shape
    temp = np.zeros((5, 5))

    final = np.zeros((l1, b1))
    pi = np.pi

    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i**2 + j**2) <= radius**2:
                temp[i, j] = 1/(C*pi*(radius**2))
            else:
                temp[i, j] = 0

    for i in range(l1-5+1):
        for j in range(b1-5+1):
            temp_box = img[i:i+5, j:j+5]
            filtered_tempbox = np.sum(temp_box*temp)
            final[i+2, j+2] = filtered_tempbox

    final = final.astype(np.uint8)
    return final

def low_pass_filter(img_, D0):
    img = img_.copy()
    l1, b1 = img.shape
    gaussian_kernel = np.zeros((l1, b1))
    # butterworth_kernel = np.zeros((l1, b1))

    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)

    for i in range(l1):
        for j in range(b1):
            abs_dist = np.sqrt((i-l1//2)**2 + (j-b1//2)**2)
            gaussian_kernel[i, j] = np.exp(-(abs_dist**2) / (2 * D0**2))
            # butterworth_kernel[i, j] = 1 / (1 + (abs_dist/D0)**2)

    # convolve_filter = fft_shift*butterworth_kernel
    convolve_filter = fft_shift*gaussian_kernel
    filtered_ifft_shift = np.fft.ifftshift(convolve_filter)
    filtered_ifft = np.fft.ifft2(filtered_ifft_shift)

    final = np.abs(filtered_ifft)
    final = final.astype(np.uint8)
    return final

def high_pass_filter(img_, D0):
    img = img_.copy()
    l1, b1 = img.shape
    gaussian_kernel = np.zeros((l1, b1))
    # butterworth_kernel = np.zeros((l1, b1))

    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)

    for i in range(l1):
        for j in range(b1):
            abs_dist = np.sqrt((i-l1//2)**2 + (j-b1//2)**2)
            gaussian_kernel[i, j] = 1 - np.exp(-(abs_dist**2) / (2 * D0**2))
            # butterworth_kernel[i, j] = 1 / (1 + (D0/abs_dist)**2)

    # convolve_filter = fft_shift*butterworth_kernel
    convolve_filter = fft_shift*gaussian_kernel
    filtered_ifft_shift = np.fft.ifftshift(convolve_filter)
    filtered_ifft = np.fft.ifft2(filtered_ifft_shift)

    final = np.abs(filtered_ifft)
    final = final.astype(np.uint8)
    return final

#Reading the image and defining it's properties
input_img = cv2.imread('cameraman.png')
l1, b1, c1 = input_img.shape

#Converting to Grayscale and padding it with zeros
gray_and_padded = GrayAndPadded(input_img, 3)

