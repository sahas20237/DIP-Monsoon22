#Genreal imports
import cv2
import numpy as np

#Reading the image and defining it's properties
input_img = cv2.imread('ruler.512_2.tiff')
l1, b1, c1 = input_img.shape
# print(l1, b1, c1)

#Making the filter
filter3x3 = np.array([(1, 1, 1), 
                      (1, 1, 1), 
                      (1, 1, 1)])*(1/9)
l2, b2 = filter3x3.shape
# print(filter3x3)
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
        padding[i+1, j+1] = gray_img[i, j]
# print(padding)

#Mean filtering
for i in range(l1):
    for j in range(b1):
        temp_box = padding[i:i+l2, j:j+b2]
        filtered_tempbox = np.sum(temp_box*filter3x3)
        gray_img[i, j] = filtered_tempbox
        
cv2.imshow('Final Output 3x3 Mean', gray_img)
cv2.waitKey(0)
# cv2.deleteallwindows()