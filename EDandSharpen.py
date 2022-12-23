#General imports
import cv2
import numpy as np
import math

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

#Sobel's Filter
sobel_vertical = np.array([(-1, 0, 1),
                           (-2, 0, 2), 
                           (-1, 0, 1)])*(1.0)

sobel_horizontal = np.array([(-1, -2, -1),
                             (0, 0, 0),
                             (1, 2, 1)])*(1.0)

#Laplacian Filter 1
laplacian_filter = np.array([(0, -1, 0), 
                             (-1, 4, -1), 
                             (0, -1, 0)])*(1.0)

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

    # Gx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    # Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    # zero = np.zeros((img.shape[0],img.shape[1]))

    # for i in range(img.shape[0]-2):
    #     for j in range(img.shape[1]-2):
    #         gx = np.sum(Gx*img[i:i + 3, j:j + 3])  
    #         gy = np.sum(Gy*img[i:i + 3, j:j + 3])
    #         zero[i + 2, j + 2] = (gx ** 2 + gy ** 2)**0.5
           
    # zero = zero.astype(np.uint8) 
    # return zero

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

#Reading the image and defining it's properties
input_img = cv2.imread('7.1.03.tiff')
l1, b1, c1 = input_img.shape

#Converting to Grayscale and padding it with zeros
gray_and_padded = GrayAndPadded(input_img, 3)

#Applying Sobel Filter
sobel = edge_detection_and_sharpen_FO(gray_and_padded, sobel_vertical, sobel_horizontal)
show_image(sobel, 'Edge-Sharpen Sobel')

#Applying Laplacian Filter
laplacian = edge_detection_and_sharpen_SO(gray_and_padded, laplacian_filter)
show_image(laplacian, 'Edge-Sharpen Laplacian')