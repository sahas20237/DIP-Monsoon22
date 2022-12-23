import math
import numpy as np
import cv2

def bicubic_spline(x, a):
    # Compute the absolute value of x
    x = np.abs(x)

    # Initialize the kernel value to 0
    y = 0

    # Compute the kernel value for different ranges of x
    if x <= 1:
        y = (a+2) * (x ** 3) - (a + 3)*(x ** 2) + 1
    elif x < 2:
        y = a * (x ** 3) - 5 * a * (x ** 2) + 8 * a * x - 4 * a
    else:
        y = 0

    # Return the spline value
    return y

def bicubic_spline_convolution(w, rel_x, rel_y, image):
    conv_value = 0
    kernel = np.zeros((w, w))
    for i in range(w):
        for j in range(w):
            kernel[i, j] = bicubic_spline(rel_x- i, 0.5) * bicubic_spline(rel_y - j, 0.1)
    conv_value = np.sum(np.multiply(image, kernel))
    return conv_value

def bicubic_spline_interpolation(im, scale):
    im = np.array(im, dtype=np.uint8)
    
    im_size = im.shape
    # get the size of the new image
    new_im_size = (int(im_size[0] * scale), int(im_size[1] * scale))
    # create a new image
    new_im = np.zeros(new_im_size)
    
    w = 3
    # Loop through the new image and compute the bicubic spline interpolation
    for i in range(new_im_size[0]):
        for j in range(new_im_size[1]):
            # Compute the exact position of this pixel in the original image
            x_exact = j / scale
            y_exact = i / scale

            # translate the exact position to the nearest pixels
            x0 = int(x_exact)
            y0 = int(y_exact)

            # check if x0 and y0 are out of bound
            if x0 + w >= im_size[0]:
                x0 = im_size[0] - w
            if y0 + w >= im_size[1]:
                y0 = im_size[1] - w
            
            # Compute the relative distances of the exact position to the nearest 4x4 pixels
            rel_x = x_exact - (x0)
            rel_y = y_exact - (y0)
            
            # Compute the convolution of the bicubic spline function with the original image
            conv_value = bicubic_spline_convolution(w, rel_x, rel_y, im[y0:y0+w, x0:x0+w])
            new_im[i, j] = conv_value
            
    # Return the new image
    new_im = new_im.astype(np.uint8)
    return new_im

#Reading the image and defining it's properties
input_img = cv2.imread('cameraman.png')
input_img = cv2.resize(input_img, (512, 512)) 
l1, b1, c1 = input_img.shape

gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

resized = gray_img[0:l1-1:int(math.sqrt(16)), 0:b1-1:int(math.sqrt(16))]
print(resized.shape)
# cv2.imshow('resized', resized)
# cv2.waitKey(0)

bc = bicubic_spline_interpolation(resized, 4)
print(bc)
cv2.imshow('bicubic', bc)
cv2.waitKey(0)