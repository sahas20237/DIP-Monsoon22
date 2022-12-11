#==============================================================================
                            # QUESTION 1
#==============================================================================

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

#==============================================================================
                            # QUESTION 2
#==============================================================================

#Genreal imports
import math
import cv2
import numpy as np

# Function to calculate PSNR
def myPSNR(original, denoised):
    mse = np.mean((original - denoised) ** 2) # Mean Squared Error
    mse = math.sqrt(mse)
    psnr = 20*math.log10(255.0/mse) # Formula taken from the net 
    return psnr

#ORDER 1 Kernels as given in the slides 
def nearestNeighbour_kernel(x):
    if abs(x) < 0.5:
        ans = 1
    else:
        ans = 0
    return ans

def bilinear_kernel(x):
    if abs(x) < 1:
        ans = 1 - abs(x)
    else:
        ans = 0
    return ans

#ORDER 2 Kernels as given in the slides 
def bell_kernel(x):
    if abs(x) < 1:
        ans = 0.75 - x**2
    elif 0.5 < abs(x) < 1.5:
        ans = 0.5 * (abs(x) - 1.5)**2
    else:
        ans = 0
    return ans

#ORDER 3 Kernels as given in the slides 
def hermite_kernel(x):
    if abs(x) <= 1:
        ans = 2 * abs(x)**3 - 3 * abs(x)**2 + 1
    else:
        ans = 0
    return ans

def bicubic_kernel(x, a):
    if abs(x) <= 1:
        ans = (a+2) * abs(x)**3 - (a+3) * abs(x)**2 + 1
    elif 1 < abs(x) < 2:
        ans = a * abs(x)**3 - 5*a * abs(x)**2 + 8*a * abs(x) - 4*a
    else:
        ans = 0
    return ans

# IMPLEMETATIONS
def nearestNeighbour(img_):
    img = img_.copy()

    final = np.zeros((l1, b1)) # Making the output array

    # Code for nearest neighbour interpolation
    for i in range(l1):
        for j in range(b1):
            final[i, j] = img[i//4, j//4] # Here 4 is taken as reduction factor = sqrt(16)

    final = final.astype(np.uint8) # Converting dtype to get the image

    return final

def bilinear(img_):
    img = img_.copy()

    l_img, b_img = img.shape
    final = np.zeros((l_img * 2, b_img * 2))

    # Code for bilinear interpolation
    for i in range(l_img):
        for j in range(b_img):

            # Here the if-else statements are used to avoid index out of bounds error
            # other statements are used to calculate the values of the pixels
            # by taking the average of the neighbouring pixels

            # the code is basic matrix multiplication and taking the average
            # just the diagonal values is divided by 4 and the edge values by 2

            final[i*2][j*2] = img[i][j]

            if(i*2 + 1 >= 2 * l_img): 
                continue
            else: 
                final[i*2 + 1][j*2] += img[i][j]//2

            if(i*2 - 1 < 0): 
                continue
            else: 
                final[i*2 - 1][j*2] += img[i][j]//2

            if(j*2 + 1 >= 2 * b_img): 
                continue
            else: 
                final[i*2][j*2+1] += img[i][j]//2

            if(j*2 - 1 < 0): 
                continue
            else: 
                final[i*2][j*2-1] += img[i][j]//2

            if(i*2 + 1 >= 2 * l_img or j*2 + 1 >= 2 * b_img): 
                continue
            else: 
                final[i*2 + 1][j*2 + 1] += img[i][j]//4

            if(i*2 + 1 >= 2 * l_img or j*2 - 1 < 0): 
                continue
            else: 
                final[i*2 + 1][j*2 - 1] += img[i][j]//4

            if(i*2 - 1 < 0 or j*2 + 1 >= 2 * b_img): 
                continue
            else: 
                final[i*2 - 1][j*2 + 1] += img[i][j]//4

            if(i*2 - 1 < 0 or j*2 - 1 < 0): 
                continue
            else: 
                final[i*2 - 1][j*2 - 1] += img[i][j]//4 

    final = final.astype(np.uint8) # Converting dtype to get the image

    return final


#Reading the image and defining it's properties
input_img = cv2.imread('cameraman.png')
# Resizing the image to 512x512 as PSNR could not be calculated with mismatched dimensions
input_img = cv2.resize(input_img, (512, 512)) 
l1, b1, c1 = input_img.shape
# print(l1, b1, c1)

#Converting image to Gray to get binary (0-255) matrix values
gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
# print(gray_img.shape)
cv2.imshow('Input Image', gray_img)
cv2.waitKey(0)

reduceFactor = 16 # Given in the question (4x4 = 16)

# Reducing the size of the image by 16 2 (4x4 both sides)
resized = gray_img[0:l1-1:int(math.sqrt(reduceFactor)), 0:b1-1:int(math.sqrt(reduceFactor))]
# print(resized.shape)
cv2.imshow('Resized Image', resized)
cv2.waitKey(0)

# OUTPUTS
nn = nearestNeighbour(resized)
# print(nn.shape)
cv2.imshow('Nearest Neighbour', nn)
cv2.waitKey(0)
print("PSNR Nearest Neighbour:", myPSNR(gray_img, nn))

bl = bilinear(bilinear(resized))
# print(bl.shape)
cv2.imshow('BiLinear', bl)
cv2.waitKey(0)
print("PSNR Bilinear:", myPSNR(gray_img, bl))