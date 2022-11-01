#General imports
import cv2
import numpy as np
import math

#Filtering Second Order
def filtering_SO(gray_img, sharp_filter):
    #Making a temprary copy of the image matrix
    temp_gray = gray_img.copy()

    #Iterating through the image matrix and 
    #adding condition so that we do not go out of range
    for i in range(l1-l2+1):
        for j in range(b1-b2+1):
            #Creating a box of size 3x3 from the image matrix
            temp_box = temp_gray[i:i+l2, j:j+b2]

            #Multiplying the box with the filter 
            filtered_tempbox = np.sum(temp_box*sharp_filter)

            #Assigning the value to the center of the box by taking the arguement
            temp_gray[i][j] = filtered_tempbox
            # temp_gray[i][j] = math.sqrt(filtered_tempbox**2 + filtered_tempbox**2)

    # After edge detection we subtract the original image to the edge detected image
    for i in range(l1):
        for j in range(b1):
            temp_gray[i][j] = temp_gray[i][j] - gray_img[i][j]
    
    return temp_gray

#Filtering First Order
def filtering_FO(gray_img, sharp_filter1, sharp_filter2):
    #Making a temprary copy of the image matrix
    temp_gray = gray_img.copy()

    #Iterating through the image matrix and 
    #adding condition so that we do not go out of range
    for i in range(l1-l2+1):
        for j in range(b1-b2+1):
            #Here we go side by side for X and Y axis

            #Creating a box of size 3x3 from the image matrix
            temp_box1 = gray_img[i:i+l2, j:j+b2]
            temp_box2 = gray_img[i:i+l2, j:j+b2]

            #Multiplying the box with the filter
            filtered_tempbox1 = np.sum(temp_box1*sharp_filter1)
            filtered_tempbox2 = np.sum(temp_box2*sharp_filter2)

            #Assigning the value to the center of the box by taking the arguement
            temp_gray[i][j] = math.sqrt(filtered_tempbox1**2 + filtered_tempbox2**2)

    #After edge detection we subtract the original image to the edge detected image
    for i in range(l1):
        for j in range(b1):
            temp_gray[i][j] = temp_gray[i][j] - gray_img[i][j]
    
    return temp_gray

#Reading the image and defining it's properties
input_img = cv2.imread('7.1.03.tiff')
l1, b1, c1 = input_img.shape

#Converting image to Gray to get binary (0-255) matrix values
gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

#Defining the filters

#Gaussian Filter
gaussian_filter = np.array([[1,4,7,4,1],
                            [4,16,26,16,4],
                            [7,26,41,26,7],
                            [4,16,26,16,4],
                            [1,4,7,4,1]])*(-1)

#Laplacian Filter 1
sharp_filter0 = np.array([(0, -1, 0), 
                          (-1, 4, -1), 
                          (0, -1, 0)])*(1)

#Laplacian Filter 2
sharp_filter1 = np.array([(-1, -1, -1), 
                          (-1, 8, -1), 
                          (-1, -1, -1)])*(1)
#Laplacian Composite
sharp_filter2 = np.array([(0, -1, 0), 
                          (-1, 5, -1), 
                          (0, -1, 0)])*(1)

#prewitt's filter
sharp_filter3_1 = np.array([(-1, 0, 1), 
                            (-1, 0, 1), 
                            (-1, 0, 1)])*(-1)

sharp_filter3_2 = np.array([(-1, -1, -1), 
                            (0, 0, 0), 
                            (1, 1, 1)])*(-1)

#sobel's filter
sharp_filter4_1 = np.array([(-1, 0, 1),
                            (-2, 0, 2), 
                            (-1, 0, 1)])*(-1)

sharp_filter4_2 = np.array([(-1, -2, -1),
                            (0, 0, 0),
                            (1, 2, 1)])*(-1)

#robert's filter
sharp_filter5_1 = np.array([(0, -1),
                            (1, 0)])*(+1)

sharp_filter5_2 = np.array([(-1, 0), 
                            (0, 1)])*(+1)

# l2, b2 = sharp_filter0.shape 
# output_img = filtering_SO(gray_img, sharp_filter0)
# cv2.imshow('Laplacian1', output_img)
# # cv2.waitKey(0)

# l2, b2 = sharp_filter1.shape
# output_img = filtering_SO(gray_img, sharp_filter1)
# cv2.imshow('Laplacian2', output_img)
# # cv2.waitKey(0)

# l2, b2 = sharp_filter2.shape
# output_img = filtering_SO(gray_img, sharp_filter2)
# cv2.imshow('Laplacian Composite', output_img)
# # cv2.waitKey(0)

# l2, b2 = sharp_filter3_1.shape
# output_img = filtering_FO(gray_img, sharp_filter3_1, sharp_filter3_2)
# cv2.imshow('Prewitt', output_img)
# # cv2.waitKey(0)

# l2, b2 = sharp_filter4_1.shape
# output_img = filtering_FO(gray_img, sharp_filter4_1, sharp_filter4_2)
# cv2.imshow('Sobel', output_img)
# # cv2.waitKey(0)

# l2, b2 = sharp_filter5_1.shape
# output_img = filtering_FO(gray_img, sharp_filter5_1, sharp_filter5_2)
# cv2.imshow('Robert', output_img)
# # cv2.waitKey(0)

#Taking a generic blur function to use in Gaussian Filtering
#We can also use our 5x5 mean filter made in Q1 to do the same
blurred = cv2.blur(gray_img, (5,5)) 
l2, b2 = gaussian_filter.shape
# output_img = filtering_Gaussian(blurred, gaussian_filter)
for i in range(l1-l2+1):
    for j in range(b1-b2+1):
        temp_gray = gray_img.copy()
        #Creating a box of size 3x3 from the image matrix
        temp_box = temp_gray[i:i+l2, j:j+b2]

        #Multiplying the box with the filter 
        filtered_tempbox = np.sum(temp_box*gaussian_filter)

        #Assigning the value to the center of the box
        temp_gray[i][j] = filtered_tempbox

# for i in range(l1):
#     for j in range(b1):
#         temp_gray[i][j] = temp_gray[i][j] - gray_img[i][j]
cv2.imshow('Gaussian', temp_gray)
cv2.waitKey(0)

cv2.destroyAllWindows()