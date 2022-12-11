#Genreal imports
import cv2
import numpy as np

#Reading the image and defining it's properties
input_img = cv2.imread('clock.tiff')
l1, b1, c1 = input_img.shape
# print(l1, b1, c1)

###############################################################################
#Explanation: 
# We are given with theta = 45* and L = 10
# So, taking a kernel of 10x10 and rotating all the ONES by 45*
# We normalize it (divide by sum of kernel) and get the following kernel

#Making the filter
kernel = np.array([(0, 0, 0, 0, 0, 0, 0, 0, 0, 1), 
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
# print(kernel)
# print(l2, b2)

##############################################################################
#Explanation: 
# We were told to do it just to be on the safer side 

#Converting image to Gray to get binary (0-255) matrix values
gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
# print(gray_img)

##############################################################################
#Explanation:
# Padding is done to avoid the loss of information at the edges
# So, taking padding of 10x10 kernal requires the following code

#Creating the padding matrix of zeros
padding = np.zeros(((l1+l2-1), (b1+b2-1)))
# print(padding)

#Inserting binary values (0-255) in padding matrix
for i in range(l1):
    for j in range(b1):
        padding[i+5, j+5] = gray_img[i, j]
# print(padding)

###############################################################################
#Explanation:
# Now, we have to do the simple convolution filter my creating a 
# temporary box of 10x10 and filter it with the kernel and then
# assign the final value to the center of the temporary box in output image

#Filtering
for i in range(l1):
    for j in range(b1):
        temp_box = padding[i:i+l2, j:j+b2]
        filtered_tempbox = np.sum(temp_box*kernel)
        gray_img[i, j] = filtered_tempbox
        
# Getting the output
cv2.imshow('Motion Blur at L=10, theta = 45*', gray_img)
cv2.waitKey(0)