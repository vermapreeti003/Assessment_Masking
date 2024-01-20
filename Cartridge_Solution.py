#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import cv2 
import numpy as np 
from matplotlib import pyplot as plt 

 


# In[2]:


# reading image 
img = cv2.imread(r'C:\Users\Preeti\Desktop\Untitled Folder\image2.png') 
output = img
result = img

# converting image into grayscale image 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

(thresh, blackAndWhiteImage) = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY) 

blur = cv2.blur(blackAndWhiteImage,(15,15))

(thresh, blackAndWhiteImage) = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY) 


# In[3]:


#show the image
cv2.imshow('Black and White', blackAndWhiteImage) 
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[4]:


#inverted image 
imageInverted = cv2.bitwise_not(blackAndWhiteImage)

contours, hierarchy = cv2.findContours(imageInverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(imageInverted, contours, -1, (255, 0, 255), 3)

cv2.imshow('Biggest Contour', imageInverted)
cv2.waitKey(0) 
cv2.destroyAllWindows()



# In[5]:


#counting all contours
cnts = sorted(contours, key=cv2.contourArea, reverse=True)

mask = np.zeros(img.shape[:2], np.uint8)
mask= cv2.bitwise_not(mask)

cv2.drawContours(mask, cnts, 1, (0,255,0), 3)
cv2.drawContours(mask, cnts, 2, (0,255,0), 3)
cv2.imshow("Mask", mask)
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[6]:


# finding 2nd and 3rd largest contours
largests = img
SecondMax= cnts[1]
thirdMax= cnts[2]
cv2.drawContours(largests, [SecondMax], 0, (0, 0, 255), 5)
cv2.drawContours(largests, [thirdMax], 0, (0, 0, 255), 5)
cv2.imshow("Mask", mask)
cv2.waitKey(0)


# In[7]:


# to fill contours finding seed 
(x,y),radius = cv2.minEnclosingCircle(SecondMax)
center = (int(x),int(y))
radius = int(radius) - 50

mask2 = np.zeros(img.shape[:2], np.uint8)
mask2 = cv2.bitwise_not(mask2)
cv2.circle(mask2,center,radius,(0,255,0),1)

h = mask2.shape[0]
w = mask2.shape[1]

seed_x=0
seed_y=0
# Find Seed
# loop over the image, pixel by pixel
for y in range(0, h):
    for x in range(0, w):
        # threshold the pixel
        if mask2[y,x] == 255:
            if mask2[y,x-1] == 0:
                seed_x = x
                seed_y = y
                break


cv2.imshow("Mask", mask)
cv2.waitKey(0)


# In[8]:


#used floodfill to fill mask
cv2.floodFill(mask, None,(seed_x, seed_y) , 0)

invertedMask = cv2.bitwise_not(mask)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
 
cv2.destroyAllWindows()


# In[9]:


# convert mask to gray and then threshold it to convert it to binary

#grayMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(invertedMask, 40, 255, cv2.THRESH_BINARY)

# split source to B,G,R channels
b,g,r = cv2.split(img)
cv2.imshow("r channel", r)
cv2.waitKey(0)
 
cv2.destroyAllWindows()


# In[ ]:


r = cv2.add(r, 30, dst = r, mask = binary, dtype = cv2.CV_8U)

cv2.merge((b,g,r), output)
cv2.imshow('Masked Image',output)
cv2.waitKey(0)
 
cv2.destroyAllWindows()


# In[ ]:




