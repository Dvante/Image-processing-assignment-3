import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

#Function responsible for adding the salt and pepper noise
def salt_and_pepper_noise(img):
    row, column = img.shape
    alteredImage = np.zeros((row, column), dtype='uint8')
    noise_density = 0.05 
    #Noise density is set to 5%
    for i in range(row):
        for j in range(column):
            random_num = random.random() 
            #Generates random number
            if random_num <noise_density/2:
                alteredImage[i,j]=0
            elif random_num <noise_density:
                alteredImage[i,j] = 255
            else:
                alteredImage[i,j] = img[i,j]
        #This for loop randomly assigns pixels in the image to be either black or white
    return alteredImage

def find_neighbors(img, row, column, filter_size):
    #The plus and minus values determine the range to search around each cell
    if filter_size==3:
        minus_value=1
        plus_value =2
    elif filter_size==5:
        minus_value=2
        plus_value = 3
    neighbor_values=[] 
    #List that stores the neighbors of the pixel
    for i in range(row-minus_value,row+plus_value):
        for j in range(column-minus_value,column+plus_value):
            if(0<=i<img.shape[0] and 0<=j<img.shape[1]):
                neighbor_values.append(img[i][j]) 
                #Adds each neighbor to the list

    return neighbor_values

def mean_filter(img, filter_size):
    row, column = img.shape
    filtered_image=np.zeros_like(img,dtype=np.uint8)
    expected_size = filter_size*filter_size
    for i in range(row):
        for j in range(column):
            neighbors = find_neighbors(img, i,j,filter_size) 
            #Gets the neighbors around the pixel
            while len(neighbors)!=expected_size:
            # pads list with zeros
                neighbors.append(0)
            totalSum = sum(neighbors) 
            #finds the sum
            average = int(totalSum/expected_size) 
            #calculate the average
            filtered_image[i][j]=average 
            #place the average in the position (i,j)
    return filtered_image

def median_filter(img, filter_size):
    row, column = img.shape
    expected_size = filter_size*filter_size
    filtered_image=np.zeros_like(img, dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            neighbors = find_neighbors(img, i,j,filter_size) 
            #Gets the neighbors for the pixel
            while len(neighbors)!=expected_size:
                neighbors.append(0) 
                #pads list with zeros
            neighbors.sort() 
            #sort the list in order to find the median
           
            #Finding the median
            if expected_size%2==0:
                m1=neighbors[expected_size//2]
                m2=neighbors[expected_size//2 -1]
                median = m1+m2 /2
            else:
                median = neighbors[expected_size//2]
            filtered_image[i][j]=median     
    return filtered_image

if __name__=='__main__':
    img = cv.imread("./Images/lena.png",-1)
    salt_and_pepper_img = salt_and_pepper_noise(img)
    print("The Images are being Prepared...")
    meanImageThree = mean_filter(salt_and_pepper_img, 3)
    medianImageThree = median_filter(salt_and_pepper_img,3)
    meanImageFive = mean_filter(salt_and_pepper_img, 5)
    medianImageFive = median_filter(salt_and_pepper_img, 5)
    
    #Displays all the images
    cv.imshow('Original Image', img)
    cv.imshow('Added Salt and Pepper', salt_and_pepper_img)
    cv.imshow('3x3 Mean Filter', meanImageThree)
    cv.imshow('3x3 Median Filter',medianImageThree)
    cv.imshow('5x5 Mean Filter', meanImageFive)
    cv.imshow('5x5 Median Filter', medianImageFive)
    cv.waitKey(0)
    
    #Plotting the images using Matplotlib
    plt.subplot(321)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    plt.gray()
    plt.subplot(322)
    plt.imshow(salt_and_pepper_img)
    plt.title('Added Salt and Pepper Noise')
    plt.axis('off')
    plt.gray()
    plt.subplot(323)
    plt.imshow(meanImageThree)
    plt.title("3x3 Mean Filter")
    plt.axis('off')
    plt.gray()
    plt.subplot(324)
    plt.imshow(medianImageThree)
    plt.title('3x3 Median Filter')
    plt.axis('off')
    plt.axis('off')
    plt.gray()
    plt.subplot(324)
    plt.imshow(medianImageThree)
    plt.title('3x3 Median Filter')
    plt.axis('off')
    plt.gray()
    plt.subplot(325)
    plt.imshow(meanImageThree)
    plt.title('5x5 Mean Filter')
    plt.axis('off')
    plt.gray()
    plt.subplot(326)
    plt.imshow(medianImageFive)
    plt.title('5x5 Median Filter')
    plt.axis('off')
    plt.gray()
    plt.show() 
