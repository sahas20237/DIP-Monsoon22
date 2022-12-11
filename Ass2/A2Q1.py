#Genreal imports
import math
import cv2
import numpy as np
import random

# Function to create Salt and Pepper Noise
def saltandpepperNoise(img_, curruptPercentage):
    img = img_.copy()

    l, b = img.shape
    totalPixels = l*b
    noisePixels = int(totalPixels*curruptPercentage/100) # Number of noise pixels 

    for _ in range(noisePixels):
        randomPixel_x = random.randint(0, l-1) # Random pixel x coordinate
        randomPixel_y = random.randint(0, b-1) # Random pixel y coordinate

        val = random.randint(0, 1) # Random value (0 or 1 for assigning WHITE or BLACK
        if val == 0:
            img[randomPixel_x][randomPixel_y] = 0
        else:
            img[randomPixel_x][randomPixel_y] = 255
    
    return img

# Median Filter as used in the previous assignment
def median5x5(img_):
    img = img_.copy()

    filter5x5 = np.array([(0, 0, 0, 0, 0), 
                          (0, 0, 0, 0, 0),
                          (0, 0, 0, 0, 0),
                          (0, 0, 0, 0, 0), 
                          (0, 0, 0, 0, 0)])
    l2, b2 = filter5x5.shape

    padding = np.zeros(((l1+l2-1), (b1+b2-1)))

    for i in range(l1):
        for j in range(b1):
            padding[i+2, j+2] = img[i, j]

    for i in range(l1):
        for j in range(b1):
            temp_box = padding[i:i+l2, j:j+b2]
            filtered_tempbox = np.median(temp_box)
            img[i, j] = filtered_tempbox

    return img

def median3x3(img_):
    img = img_.copy()

    filter3x3 = np.array([(0, 0, 0),
                          (0, 0, 0),
                          (0, 0, 0)])
    l2, b2 = filter3x3.shape

    padding = np.zeros(((l1+l2-1), (b1+b2-1)))

    for i in range(l1):
        for j in range(b1):
            padding[i+1, j+1] = img[i, j]

    for i in range(l1):
        for j in range(b1):
            temp_box = padding[i:i+l2, j:j+b2]
            filtered_tempbox = np.median(temp_box)
            img[i, j] = filtered_tempbox

    return img

# Function to calculate PSNR
def myPSNR(original, denoised):
    mse = np.mean((original - denoised) ** 2) # Mean Squared Error
    mse = math.sqrt(mse)
    psnr = 20*math.log10(255.0/mse) # Formula taken from the net 
    return psnr

#Reading the image and defining it's properties
input_img = cv2.imread('barbara_gray.bmp')
l1, b1, c1 = input_img.shape
# print(l1, b1, c1)

#Converting image to Gray to get binary (0-255) matrix values
gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Input Image', gray_img)
cv2.waitKey(0)

#OUTPUTS
output_img = saltandpepperNoise(gray_img, 5)
denoised_img = median5x5(output_img)
cv2.imshow('5% noise and denoised', np.concatenate((output_img, denoised_img), axis=1))
cv2.waitKey(0)
mypsnr = myPSNR(gray_img, denoised_img)
print("PSNR 5%:", mypsnr)

output_img = saltandpepperNoise(gray_img, 15)
denoised_img = median5x5(output_img)
cv2.imshow('15% noise and denoised', np.concatenate((output_img, denoised_img), axis=1))
cv2.waitKey(0)
mypsnr = myPSNR(gray_img, denoised_img)
print("PSNR 15%:", mypsnr)

output_img = saltandpepperNoise(gray_img, 20)
denoised_img = median5x5(output_img)
cv2.imshow('20% noise and denoised', np.concatenate((output_img, denoised_img), axis=1))
cv2.waitKey(0)
mypsnr = myPSNR(gray_img, denoised_img)
print("PSNR 20%:", mypsnr)

output_img = saltandpepperNoise(gray_img, 25)
denoised_img = median5x5(output_img)
cv2.imshow('25% noise and denoised', np.concatenate((output_img, denoised_img), axis=1))
cv2.waitKey(0)
mypsnr = myPSNR(gray_img, denoised_img)
print("PSNR 25%:", mypsnr)

cv2.destroyAllWindows()