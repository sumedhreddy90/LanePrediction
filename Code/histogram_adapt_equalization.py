import sys
import math
import os
from os.path import isfile, join
from tkinter import Image
import numpy as np
import cv2
import scipy
from scipy import fft, ifft
from numpy import histogram_bin_edges, linalg as LA
import matplotlib.pyplot as plt
from PIL import Image 

def Histogram(input):
    histogram_list = list()
    
    for i in range(256):
        search_idx = np.where(input == i)
        histogram_list.append( [ i , len(search_idx[0]) ] )
        
    return histogram_list

def histogramEqualization(frame):

    # Appling BGR2HSV on each frame
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Extracting V Channel
    v_channel = image_hsv[:,:,2]
    height,width = v_channel.shape   
    histogram = Histogram(v_channel)

    # normalizing by N pixels with intessities <= i
    cumulative_dist = list()
    zx = 0
    for i in range(len(histogram)):
        zx = zx + (histogram[i][1]/(height * width))
        cumulative_dist.append(round(zx*255))
    new_histogram = np.asarray(cumulative_dist)
    
    # Image processing via equalization
    image_hsv[:,:,2] = new_histogram[image_hsv[:,:,2]] 
    
    hsv_bgr_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    
    return histogram, hsv_bgr_image

def adaptiveHistogramEqualization(frame):
    # Splitting images in tiles 5*5
    # Applying histogram for every tile
    img_height=int(frame.shape[0]/5)
    img_width=int(frame.shape[1]/5)
    for i in range(0,5):
        for j in range(0,5):
            tile = frame[int(i*img_height):int((i+1)*img_height),int(j*img_width):int((j+1)*img_width)]
            _, adaptive_tile = histogramEqualization(tile)
            frame[int(i*img_height):int((i+1)*img_height),int(j*img_width):int((j+1)*img_width)]= adaptive_tile

    return frame
# Video Generating function
def generate_video():
    image_folder = './Data/' # make sure to use your folder
    video_name = 'input_problem1.avi'
    os.chdir("/Users/sumedhreddy/Desktop/LanePrediction/")
      
    images = [frame for frame in os.listdir(image_folder)
              if frame.endswith(".jpg") or
                 frame.endswith(".jpeg") or
                 frame.endswith("png")]
     
    # Array images should only consider
    # the image files ignoring others if any
  
    frame = cv2.imread(os.path.join(image_folder, images[0]))
  
    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape  
  
    video = cv2.VideoWriter(video_name, 0, 1, (width, height)) 
  
    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(image_folder, image))) 
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release() 
    
     
# Implementation of Histogram equalization - Normal and Adaptive
# Program to enhance the contrast and improve the visual appearance of the video sequence

# Problem 1 a: Normal historgram equalization

# Folder which contains all the images
# from which video is to be generated
os.chdir("/Users/sumedhreddy/Desktop/LanePrediction/Data")  
path = "/Users/sumedhreddy/Desktop/LanePrediction/Data" # set path depending on your OS
  
mean_height = 0
mean_width = 0
  
num_of_images = len(os.listdir('.'))
print('Total number of images in the Folder:',num_of_images)
  
for file in os.listdir('.'):
    im = Image.open(os.path.join(path, file))
    width, height = im.size
    mean_width += width
    mean_height += height
  
# Finding the mean height and width of all images.
# This is required because the video frame needs
# to be set with same width and height. Otherwise
# images not equal to that width height will not get 
# embedded into the video
mean_width = int(mean_width / num_of_images)
mean_height = int(mean_height / num_of_images)
  
# print(mean_height)
# print(mean_width)
  
# Resizing of the images to give
# them same width and height 
for file in os.listdir('.'):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
        # opening image using PIL Image
        im = Image.open(os.path.join(path, file)) 
   
        # im.size includes the height and width of image
        width, height = im.size   
  
        # resizing 
        imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS) 
        imResize.save( file, 'JPEG', quality = 95) # setting quality
        # printing each resized image name
        # print(im.filename.split('\\')[-1], " is resized") 


# Calling the generate_video function
print("Processing Images into video")
generate_video()

input_=cv2.VideoCapture('input_problem1.avi')
input_.set(1,1)

output_histogram = cv2.VideoWriter("histogram_problem_1a.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (1224, 370))
adapt_histogram = cv2.VideoWriter("adaptive_problem_1b.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (1224, 370))

if (input_.isOpened() == False):
    print('Error opening the file!')

count = 0
while (input_.isOpened()):
    count+=1
    success, image = input_.read()
    
    if success:
        print("Performing Histogram equalization")
        img = image.copy()
        img1 = image.copy()
        histogram, histogram_image=histogramEqualization(img)
        for i in range(0,24):
            output_histogram.write(histogram_image)
        print("Performing Adaptive Histogram equalization")
        hist_adapt = adaptiveHistogramEqualization(img1)
        for i in range(0,24):
            adapt_histogram.write(hist_adapt)

    if count == 5:
        print("Applying and Displaying Histogram on Frame 5")
        plt.hist(histogram_image.ravel(),256,[0,256])
        plt.savefig('histogram_plot.png')
        print("Histogram Graph saved as: ", 'histogram_plot.png')
        # Comparing input image with histogram image
        compare_images = np.vstack((image, histogram_image))
        cv2.imwrite("Histogram_frame.jpg" , histogram_image)
        cv2.imwrite("Histogram_Compare.jpg", compare_images)
        print("Histogram Equalization Complete!!!'")
        ########################################################
        # Problem 1b: Adaptive historgram equalization
        plt.hist(hist_adapt.ravel(),256,[0,256])
        plt.savefig('Adaptive_histogram_plot.png')
        print("Histogram Frame 5 saved as: ", 'Adaptive_histogram_plot.png')
        # Comparing input image with histogram image
        compare_images = np.vstack((image, histogram_image, hist_adapt))
        cv2.imwrite("Adaptive_Histogram_frame.jpg" , hist_adapt)
        cv2.imwrite("Adaptive_Histogram_Compare.jpg", compare_images)
        print("Adaptive Histogram Equalization Complete!!!'")
        break

input_.release()
adapt_histogram.release()
output_histogram.release()
cv2.destroyAllWindows()