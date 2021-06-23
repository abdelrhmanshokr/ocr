import numpy as np 
import cv2 
import matplotlib.pyplot as plt 


imageSrc = cv2.imread('test.jpeg')

# First cut the source down slightly
h = imageSrc.shape[0]
w = imageSrc.shape[1]
cropInitial = 10
imageSrc = imageSrc[100:50+(h-cropInitial*2), 50:50+(w-cropInitial*2)]

# Threshold the image and find edges (to reduce the amount of pixels to count)
ret, imageDest = cv2.threshold(imageSrc, 180, 255, cv2.THRESH_BINARY_INV)
imageDest = cv2.Canny(imageDest, 100, 100, 3)

# Create a list of remaining pixels
points = cv2.findNonZero(imageDest)

# Calculate a bounding rectangle for these points
hull = cv2.convexHull(points)
x,y,w,h = cv2.boundingRect(hull)

# Crop the original image to the bounding rectangle
imageResult = imageSrc[y:y+h,x:x+w]
cv2.imwrite('cropped_result.jpeg', imageResult)
cv2.imshow('imageResult.jpeg', imageResult)
cv2.waitKey(0)
