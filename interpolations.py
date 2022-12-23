#Genreal imports
import math
import cv2
import numpy as np

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

def bicubic(img_, scalingFac):
    img = img_.copy()
    l1, b1 = img.shape

    final = np.zeros((l1 * scalingFac, b1 * scalingFac), dtype = int)
    l2, b2 = final.shape

    for i in range(l2):
        for j in range(b2):
            x_exact = j/scalingFac 
            y_exact = i/scalingFac

            x0 = int(x_exact)
            y0 = int(y_exact)

            if x0 + 3 >= l1:
                x0 = l1 - 3
            if y0 + 3 >= b1:
                y0 = b1 - 3

            x_rel = x_exact - x0
            y_rel = y_exact - y0

            kernel = np.zeros((3, 3))
            ans = 0
            temp_box = img[y0:y0+3, x0:x0+3]

            for i in range(3):
                for j in range(3):
                    kernel[i, j] = bicubic_kernel(x_rel-i, 0.5) * bicubic_kernel(y_rel-j, 0.1)
            ans = np.sum(temp_box*kernel)
            
            print(kernel)
            
            final[i, j] = ans

    final = final.astype(np.uint8)
    return final

#Reading the image and defining it's properties
input_img = cv2.imread('cameraman.png')
input_img = cv2.resize(input_img, (512, 512)) 
l1, b1, c1 = input_img.shape

gray_and_padded = GrayAndPadded(input_img, 3)

scalingFactor = 16 # Given in the question (4x4 = 16)

# Reducing the size of the image by 16 2 (4x4 both sides)
resized = gray_and_padded[0:l1-1:int(math.sqrt(scalingFactor)), 0:b1-1:int(math.sqrt(scalingFactor))]
# print(resized.shape)
# show_image(resized, 'Resized')

# OUTPUTS
# nn = nearestNeighbour(resized)
# show_image(nn, 'Nearest Neighbour')

# bl = bilinear(bilinear(resized))
# show_image(bl, 'Bilinear')

# bc = bicubic(resized, 4)
# print(bc)
# show_image(bc, 'Bicubic')