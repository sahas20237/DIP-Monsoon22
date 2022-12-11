#Genreal imports
import math
import random
import cv2
import numpy as np

#Reading the image and defining it's properties
input_img = cv2.imread('')
l1, b1, c1 = input_img.shape
# print(l1, b1, c1)

#Making the filter
kernel = np.zeros((3, 3)) # change here for filter size
l2, b2 = kernel.shape
# print(kernel)
# print(l2, b2)

#Converting image to Gray to get binary (0-255) matrix values
gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
# print(gray_img)
# print(len(gray_img), len(gray_img[0]))
# cv2.imshow('Gray Image', gray_img)
# cv2.waitKey(0)

#Creating the padding matrix of zeros
padding = np.zeros(((l1+l2-1), (b1+b2-1)))
# print(padding)

#Inserting binary values (0-255) in padding matrix
for i in range(l1):
    for j in range(b1):
        padding[i+1, j+1] = gray_img[i, j] # change here according to filter size
# print(padding)

#Mean filtering
for i in range(l1):
    for j in range(b1):
        temp_box = padding[i:i+l2, j:j+b2]
        filtered_tempbox = np.sum(temp_box*kernel)
        gray_img[i, j] = filtered_tempbox
        
cv2.imshow('Final Output', gray_img)
cv2.waitKey(0)

# cv2.deleteallwindows()