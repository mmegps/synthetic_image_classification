import numpy as np
import cv2
import os
 
dirpath = os.getcwd()
IMAGE_SIZE = 256

DATADIR = "./train_images"
CATEGORIES = ["c0", "c1", "c2"]

for j in range(1,15000):
    # Create a black image
    img = np.zeros((IMAGE_SIZE,IMAGE_SIZE,3), np.uint8)
    img.fill(255) # or img[:] = 255

    complexity = np.random.randint(1, 30, size=1)[0]
    print (complexity)

    imagefolder = "train_images/"
    if complexity <= 10:
        imagefolder = imagefolder + "c0"
    elif complexity <= 20:
        imagefolder = imagefolder + "c1"
    else:
        imagefolder = imagefolder + "c2"

    for i in range(complexity):
        startCoord = np.random.randint(0, IMAGE_SIZE, size=2)
        endCoord   = np.random.randint(0, IMAGE_SIZE, size=2)

        # Draw a random blue line with thickness of 2 px
        #todo: Could vary thickness, background colour
        cv2.line(img,(startCoord[0],startCoord[1]), (endCoord[0], endCoord[1]), (255,0,0),2)

    #cv2.imshow('image',img)
    filelocation = dirpath + "/" + imagefolder + "/line" + str(j) +".png"

    cv2.imwrite(filelocation,img)