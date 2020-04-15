"""
A library used for automatic MMPI test checking
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

def grayscaleThreshold(image, threshold=128, resultingType = "uint8"):
    """
    Thresholding function
    """
    return ((image > threshold) * 255).astype(resultingType)

def loadModel(path):
    
    from keras.models import Sequential
    from keras.layers import Activation,Dense, Conv2D, Dropout, Flatten, MaxPooling2D
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.normalization import BatchNormalization

    #load the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(35,35,1),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(Dense(2, activation='softmax'))
    model.load_weights(path)
    
    return model

def orderPoints(points):
    """
    A function that takes four tuples of unordered points and
    returns a Numpy array of ordered points: 
    [1] top left
    [2] top right
    [3] bottom right
    [4] bottom left
    """
    
    orderedPoints = np.zeros((4, 2), dtype = "float32")
    # compute top left and bottom right points
    s = points.sum(axis = 1)
    orderedPoints[0] = points[np.argmin(s)]
    orderedPoints[2] = points[np.argmax(s)]

    # compute top right and bottom left points
    difference = np.diff(points, axis = 1)
    orderedPoints[1] = points[np.argmin(difference)]
    orderedPoints[3] = points[np.argmax(difference)]

    return orderedPoints

def findFourSquares(image):
    # may be useful to add image resize to apply morpholody
    cv2.imwrite("preThchecker.jpg", image)
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    retval, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    img = 255 - img
    img = cv2.erode(img, np.ones((23,23)))
    cv2.imwrite("postThchecker.jpg", img)
    retval, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    img = cv2.dilate(img, np.ones((23,23)))
    cv2.imwrite("postMorphchecker.jpg", img)
    cnts,_ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    locs=[]
    centroids = []

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # aspect ratio
        if ar > 0.5 and ar < 2:
            if (w > 20) and (h > 20) and (w < 150) and (h < 150):
                locs.append((x, y, w, h))
                centroids.append((x+w/2, y+h/2))
                cv2.rectangle(img, (x-5, y - 5), (x + w + 5, y + h + 5), (255, 255, 255), 2)

    if len(centroids) != 4:
        print("Centroids: Something is wrong. There are " + str(len(centroids)) + " centroids")
    
    centroids = np.array(centroids)
    
    return centroids

def formTransformation(image):
    """
    A function that transforms the raw image to needed width and height
    based on four marker points. Takes the image and points loctions
    as inputs.
    """
    points = findFourSquares(image)
    goodPoints = orderPoints(points)
    (topLeft, topRight, bottomRight, bottomLeft) = goodPoints

    maximumWidth = 1600
    maximumHeight = 1600

    # Specifying destination points of transformation
    destination = np.array([
        [0, 0],
        [maximumWidth  - 1, 0],
        [maximumWidth  - 1, maximumHeight - 1],
        [0, maximumHeight - 1]], dtype = "float32")

    # Computing the transformation matrix
    transformationMatrix = cv2.getPerspectiveTransform(goodPoints, destination)
    
    # Warp the image
    warpedImage = cv2.warpPerspective(image, transformationMatrix, (maximumWidth , maximumHeight))

    return warpedImage

def grayscaleThreshold(image, threshold=128, resultingType = "uint8"):
    """
    Thresholding function
    """
    return ((image > threshold) * 255).astype(resultingType)

def paddington(img, wid, hei, centerImage = False):
    """
    Padding function
    
    Adding padding 
    image = cv2.copyMakeBorder( src, top, bottom, left, right, borderType)
    
    Saving the image
    cv2.imwrite('',gray[(ypt-5):(ypt+hei+5), (xpt-5):(xpt+wid+5)])
    
    """
    
    #conversion to grayscale if necessary
    
    if len(np.shape(img)) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    binaryImg = grayscaleThreshold(img, 150)
        
    padded_bear = img
    #adding padding to full square
    if np.shape(img)[0] > np.shape(img)[1]:
        padded_bear = cv2.copyMakeBorder(img, 0, 0, int((np.shape(img)[0]-np.shape(img)[1])/2), int((np.shape(img)[0]-np.shape(img)[1])/2), cv2.BORDER_CONSTANT,value=[255,255,255])
    elif np.shape(img)[0] < np.shape(img)[1]:
        padded_bear = cv2.copyMakeBorder(img, int((np.shape(img)[1]-np.shape(img)[0])/2), int((np.shape(img)[1]-np.shape(img)[0])/2), 0, 0, cv2.BORDER_CONSTANT,value=[255,255,255])
    
    if np.shape(padded_bear)[0] > np.shape(padded_bear)[1]:
        padded_bear = cv2.copyMakeBorder(padded_bear, 0, 0, int((np.shape(padded_bear)[0]-np.shape(padded_bear)[1])), 0, cv2.BORDER_CONSTANT,value=[255,255,255])
    elif np.shape(padded_bear)[0] < np.shape(padded_bear)[1]:
        padded_bear = cv2.copyMakeBorder(padded_bear, int((np.shape(padded_bear)[1]-np.shape(padded_bear)[0])), 0, 0, 0, cv2.BORDER_CONSTANT,value=[255,255,255])
        
    return padded_bear

def ROIextractor(model, image):
    """
    A function needed to extract the Region Of Interest
    """
    count = 0
    orig = image

    #work on morphology to remove noise
    if len(np.shape(image)) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)[1]
    image = 255 - image
    
    #find contours
    output = []

    letter_counts = np.zeros((3,2))
    for i in range(3):
        letter_counts[i,0] = i

    ans_locs = []
    checkerx=[66,126,182,244,296,360,415, 478, 533,593, 647,709, 763,825,880,942,997,1058,1112,1173,1228,1290,1344,1406,1462,1524]
    checkery=[60,103,146,189,233,311,355,398,442,485,564,607,651,694,737,815,856,911,954,997,1076,1118,1163,1207,1250,1329,1372,1416,1459,1503]
     
    larr = []
    for i in range(26):
        for j in range(30):
            if (checkerx[i] > 1410 and checkery[j] > 900):
                pass
            else:
                larr.append([checkerx[i],checkery[j]])
    larr = np.array(larr)
   
    locs = np.zeros((377*2, 4))

    for i in range(377*2):
        locs[i,0] = int(larr[i,0])
        locs[i,1] = int(larr[i,1])
        locs[i,2] = int(35)
        locs[i,3] = int(35)

    # loop over the contours to label them
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # initialize the list of group digits
        groupOutput = []
        gX = int(gX)
        gY = int(gY)
        gW = int(gW)
        gH = int(gH)
        # extract the group ROI of 4 digits from the grayscale image,
        # then apply thresholding to segment the digits from the
        # background of the credit card
        group = image[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #plti(group,cmap='Greys')


        bear = paddington(orig[(gY-5):(gY+gH+5), (gX-5):(gX+gW+5)], gW, gH)
        bear = cv2.resize(bear, (35, 35), interpolation = cv2.INTER_AREA)
        bear = cv2.threshold(bear,127,255,cv2.THRESH_BINARY)[1]
        bear = bear - 255
        pred = model.predict(bear.reshape(1, 35, 35, 1))
        scr = pred.argmax()
        for val in range(3):
            if val == np.int64(scr):
                letter_counts[val,1] = letter_counts[val,1] + 1
                if val == 0 or val == 1:
                    count = count + 1
                    ans_locs.append((gX, gY, gW, gH,val,count))
                    cv2.rectangle(orig, (gX-5, gY - 5), (gX + gW + 5, gY + gH + 5), (255, 0, 255), 2)
                    cv2.putText(orig,str(np.int64(val)),(gX,gY), cv2.FONT_HERSHEY_SIMPLEX , 0.5,(255,0,0),2,cv2.LINE_AA)
                else:
                    cv2.rectangle(orig, (gX-5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 255, 0), 2)
    cv2.imwrite("checker.jpg", orig)
    print()
    return ans_locs


def locationSorting(ans_locs):
    f01t30t = []
    f31t60t = []
    f61t90t = []
    f91t120t = []
    f121t150t = []
    f151t180t = []
    f181t210t = []
    f211t240t = []
    f241t270t = []
    f271t300t = []
    f301t330t = []
    f331t360t = []
    f361t377t = []

    f01t30f = []
    f31t60f = []
    f61t90f = []
    f91t120f = []
    f121t150f = []
    f151t180f = []
    f181t210f = []
    f211t240f = []
    f241t270f = []
    f271t300f = []
    f301t330f = []
    f331t360f = []
    f361t377f = []

    #sorting and splitting by x-position
    for (i, (gX, gY, gW, gH,val,count)) in enumerate(ans_locs):
        #1-30
        if count <= 30:
            f01t30t.append((gX, gY, gW, gH,val,count))
        elif 31 <= count <= 60:
            f01t30f.append((gX, gY, gW, gH,val,count))
        #31-60
        elif 61 <= count <= 90:
            f31t60t.append((gX, gY, gW, gH,val,count))
        elif 91 <= count <= 120:
            f31t60f.append((gX, gY, gW, gH,val,count))
        #61-90
        elif 121 <= count <= 150:
            f61t90t.append((gX, gY, gW, gH,val,count))
        elif 151 <= count <= 180:
            f61t90f.append((gX, gY, gW, gH,val,count))
        #91-120    
        elif 181 <= count <= 210:
            f91t120t.append((gX, gY, gW, gH,val,count))
        elif 211 <= count <= 240:
            f91t120f.append((gX, gY, gW, gH,val,count))
        #121-150
        elif 241 <= count <= 270:
            f121t150t.append((gX, gY, gW, gH,val,count))
        elif 271 <= count <= 300:
            f121t150f.append((gX, gY, gW, gH,val,count))
        #151-180
        elif 301 <= count <= 330:
            f151t180t.append((gX, gY, gW, gH,val,count))
        elif 331 <= count <= 360:
            f151t180f.append((gX, gY, gW, gH,val,count))
        #181-210
        elif 361 <= count <= 390:
            f181t210t.append((gX, gY, gW, gH,val,count))
        elif 391 <= count <= 420:
            f181t210f.append((gX, gY, gW, gH,val,count))
        #211-240   
        elif 421 <= count <= 450:
            f211t240t.append((gX, gY, gW, gH,val,count))
        elif 451 <= count <= 480:
            f211t240f.append((gX, gY, gW, gH,val,count))
        #241-270
        elif 481 <= count <= 510:
            f241t270t.append((gX, gY, gW, gH,val,count))
        elif 511 <= count <= 540:
            f241t270f.append((gX, gY, gW, gH,val,count))
        #271-300
        elif 541 <= count <= 570:
            f271t300t.append((gX, gY, gW, gH,val,count))
        elif 571 <= count <= 600:
            f271t300f.append((gX, gY, gW, gH,val,count))   
        #301-330
        elif 601 <= count <= 630:
            f301t330t.append((gX, gY, gW, gH,val,count))
        elif 631 <= count <= 660:
            f301t330f.append((gX, gY, gW, gH,val,count))
        #331-360
        elif 661 <= count <= 690:
            f331t360t.append((gX, gY, gW, gH,val,count))
        elif 691 <= count <= 720:
            f331t360f.append((gX, gY, gW, gH,val,count))  
        #361-377
        elif 721 <= count <= 737:
            f361t377t.append((gX, gY, gW, gH,val,count))
        elif 738 <= count <= 754:
            f361t377f.append((gX, gY, gW, gH,val,count))    
        else:
            #continue
            print("Invalid number of boxes.")    

    #sorting lists by y-position
    f01t30t = sorted(f01t30t, key=lambda x:x[1])
    f31t60t = sorted(f31t60t, key=lambda x:x[1])
    f61t90t = sorted(f61t90t, key=lambda x:x[1])
    f91t120t = sorted(f91t120t, key=lambda x:x[1])
    f121t150t = sorted(f121t150t, key=lambda x:x[1])
    f151t180t = sorted(f151t180t, key=lambda x:x[1])
    f181t210t = sorted(f181t210t, key=lambda x:x[1])
    f211t240t = sorted(f211t240t, key=lambda x:x[1])
    f241t270t = sorted(f241t270t, key=lambda x:x[1])
    f271t300t = sorted(f271t300t, key=lambda x:x[1])
    f301t330t = sorted(f301t330t, key=lambda x:x[1])
    f331t360t = sorted(f331t360t, key=lambda x:x[1])
    f361t377t = sorted(f361t377t, key=lambda x:x[1])

    f01t30f = sorted(f01t30f, key=lambda x:x[1])
    f31t60f = sorted(f31t60f, key=lambda x:x[1])
    f61t90f = sorted(f61t90f, key=lambda x:x[1])
    f91t120f = sorted(f91t120f, key=lambda x:x[1])
    f121t150f = sorted(f121t150f, key=lambda x:x[1])
    f151t180f = sorted(f151t180f, key=lambda x:x[1])
    f181t210f = sorted(f181t210f, key=lambda x:x[1])
    f211t240f = sorted(f211t240f, key=lambda x:x[1])
    f241t270f = sorted(f241t270f, key=lambda x:x[1])
    f271t300f = sorted(f271t300f, key=lambda x:x[1])
    f301t330f = sorted(f301t330f, key=lambda x:x[1])
    f331t360f = sorted(f331t360f, key=lambda x:x[1])
    f361t377f = sorted(f361t377f, key=lambda x:x[1])

    #concatenating total arrays
    allfather_t=f01t30t+f31t60t+f61t90t+f91t120t+f121t150t+f151t180t+f181t210t+f211t240t+f241t270t+f271t300t+f301t330t+f331t360t+f361t377t
    allfather_f=f01t30f+f31t60f+f61t90f+f91t120f+f121t150f+f151t180f+f181t210f+f211t240f+f241t270f+f271t300f+f301t330f+f331t360f+f361t377f

    true = np.array(allfather_t)[:,4]
    false = np.array(allfather_f)[:,4]
    
    return true, false

def rawScoreCounter(true, false):
    #computing scale values
    scale_L_f = [50, 58, 65, 90, 120, 150, 163, 180, 210, 231, 240, 270, 300, 330, 360]
    scale_L_t = []

    scale_F_f = [24, 57, 58, 84, 88, 176, 193, 233, 235, 261, 263, 276, 296, 323, 364]
    scale_F_t = [12, 25, 26, 27, 28, 54, 55, 56, 72, 83, 85, 86, 102, 105, 113, 115, 116, 117, 132, 143, 145, 146, 147, 173, 175, 177, 203, 206, 207, 236, 237, 265, 266, 267, 294, 295, 297, 324, 325, 326, 327, 334, 353, 354, 355, 356, 357]

    scale_K_f = [8, 13, 38, 43, 73, 94, 98, 103, 124, 128, 133, 154, 158, 163, 188, 193, 217, 223, 253, 277, 280, 282, 283, 310, 312, 313, 342, 372]
    scale_K_t = [340]

    scale_1_f = [16, 47, 75, 167, 195, 254, 284, 374]
    scale_1_t = [15, 17, 45, 46, 77, 105, 107, 135, 137, 165, 197, 225, 255, 285, 286, 308, 314, 315, 316, 344, 345, 346, 375, 376]

    scale_2_f = [18, 20, 41, 43, 50, 75, 78, 131, 137, 138, 161, 163, 167, 193, 198, 199, 223, 227, 254, 277, 284, 287, 288, 289, 317, 318, 319, 338, 347, 348, 349, 368, 370, 377]
    scale_2_t = [9, 19, 48, 49, 98, 105, 108, 109, 139, 165, 168, 169, 225, 228, 229, 253, 257, 258, 259, 315, 337, 367]

    scale_3_f = [8, 11, 13, 16, 41, 43, 71, 73, 74, 75, 101, 103, 104, 124, 133, 155, 163, 164, 184, 187, 196, 214, 218, 224, 226, 248, 254, 256, 278, 280, 284, 343, 370, 374]
    scale_3_t = [14, 15, 45, 46, 76, 105, 106, 134, 135, 136, 165, 166, 194, 225, 255, 285, 314, 315, 344, 345, 373, 375]

    scale_4_f = [8, 10, 11, 38, 41, 68, 71, 94, 101, 130, 131, 160, 161, 187, 217, 220, 277, 280, 307, 310, 340, 370]
    scale_4_t = [12, 40, 42, 64, 70, 72, 100, 102, 132, 162, 190, 191, 192, 221, 222, 247, 250, 251, 252, 281, 311, 337, 341, 366, 367, 369, 371]

    scale_5_f = [2, 4, 31, 33, 34, 35, 61, 63, 65, 91, 92, 121, 123, 124, 153, 182, 183, 184, 211, 212, 214, 241, 244, 271, 272, 304, 333, 363, 364]
    scale_5_t = [1, 3, 5, 32, 62, 93, 94, 122, 151, 152, 154, 181, 213, 242, 243, 273, 274, 301, 302, 303, 331, 332, 334, 361, 362]

    scale_6_f = [34, 117, 148, 188, 196, 218, 226, 238, 268, 370]
    scale_6_t = [5, 12, 28, 42, 51, 88, 113, 114, 143, 144, 162, 171, 178, 192, 203, 208, 222, 231, 252, 259, 262, 267, 291, 297, 308, 327, 339, 353, 371]

    scale_7_f = [41, 195, 200, 288, 318, 348]
    scale_7_t = [19, 21, 39, 49, 51, 69, 76, 79, 80, 81, 99, 106, 109, 110, 111, 136, 140, 141, 154, 159, 170, 171, 189, 191, 201, 219, 221, 230, 231, 251, 253, 258, 260, 290, 291, 315, 320, 337, 350, 367]

    scale_8_f = [24, 41, 84, 248, 263, 283, 292, 293, 322, 323, 348]
    scale_8_t = [12, 21, 22, 23, 42, 51, 52, 53, 54, 79, 81, 82, 83, 106, 109, 111, 112, 113, 114, 136, 139, 141, 142, 143, 144, 167, 169, 171, 172, 173, 174, 201, 202, 203, 204, 247, 274, 279, 304, 308, 309, 311, 321, 337, 341, 345, 350, 351, 352, 353, 371, 375]

    scale_9_f = [8, 30, 35, 38, 71, 89, 90, 120, 217, 249, 313, 358]
    scale_9_t = [20, 21, 29, 51, 59, 60, 94, 105, 108, 119, 149, 174, 179, 204, 209, 222, 234, 239, 256, 262, 264, 269, 276, 281, 289, 298, 319, 328, 339, 349, 353, 359]

    scale_0_f = [4, 36, 66, 67, 68, 96, 125, 156, 157, 185, 186, 189, 216, 246, 249, 273, 275, 276, 277, 303, 333, 335, 336, 339, 363, 368]
    scale_0_t = [6, 7, 8, 9, 34, 37, 38, 39, 69, 95, 97, 98, 126, 127, 128, 129, 155, 158, 159, 187, 188, 217, 218, 219, 243, 245, 248, 278, 279, 307, 308, 309, 337, 338, 365, 366, 367]

    #decrementing every element by 1, as arrays start from 0
    scale_L_f = np.int64(scale_L_f-np.ones(np.shape(scale_L_f)))

    scale_F_f = np.int64(scale_F_f-np.ones(np.shape(scale_F_f)))
    scale_F_t = np.int64(scale_F_t-np.ones(np.shape(scale_F_t)))

    scale_K_f = np.int64(scale_K_f-np.ones(np.shape(scale_K_f)))
    scale_K_t = np.int64(scale_K_t-np.ones(np.shape(scale_K_t)))

    scale_1_f = np.int64(scale_1_f-np.ones(np.shape(scale_1_f)))
    scale_1_t = np.int64(scale_1_t-np.ones(np.shape(scale_1_t)))

    scale_2_f = np.int64(scale_2_f-np.ones(np.shape(scale_2_f)))
    scale_2_t = np.int64(scale_2_t-np.ones(np.shape(scale_2_t)))

    scale_3_f = np.int64(scale_3_f-np.ones(np.shape(scale_3_f)))
    scale_3_t = np.int64(scale_3_t-np.ones(np.shape(scale_3_t)))

    scale_4_f = np.int64(scale_4_f-np.ones(np.shape(scale_4_f)))
    scale_4_t = np.int64(scale_4_t-np.ones(np.shape(scale_4_t)))

    scale_5_f = np.int64(scale_5_f-np.ones(np.shape(scale_5_f)))
    scale_5_t = np.int64(scale_5_t-np.ones(np.shape(scale_5_t)))

    scale_6_f = np.int64(scale_6_f-np.ones(np.shape(scale_6_f)))
    scale_6_t = np.int64(scale_6_t-np.ones(np.shape(scale_6_t)))

    scale_7_f = np.int64(scale_7_f-np.ones(np.shape(scale_7_f)))
    scale_7_t = np.int64(scale_7_t-np.ones(np.shape(scale_7_t)))

    scale_8_f = np.int64(scale_8_f-np.ones(np.shape(scale_8_f)))
    scale_8_t = np.int64(scale_8_t-np.ones(np.shape(scale_8_t)))

    scale_9_f = np.int64(scale_9_f-np.ones(np.shape(scale_9_f)))
    scale_9_t = np.int64(scale_9_t-np.ones(np.shape(scale_9_t)))

    scale_0_f = np.int64(scale_0_f-np.ones(np.shape(scale_0_f)))
    scale_0_t = np.int64(scale_0_t-np.ones(np.shape(scale_0_t)))


    #counting raw scores

    scale_L = np.sum(false[scale_L_f]) + np.sum(true[scale_L_t])
    scale_F = np.sum(false[scale_F_f]) + np.sum(true[scale_F_t])
    scale_K = np.sum(false[scale_K_f]) + np.sum(true[scale_K_t])

    scale_1 = np.int64(np.sum(false[scale_1_f]) + np.sum(true[scale_1_t]) + np.ceil(0.5*scale_K))
    scale_2 = np.int64(np.sum(false[scale_2_f]) + np.sum(true[scale_2_t]))
    scale_3 = np.int64(np.sum(false[scale_3_f]) + np.sum(true[scale_3_t]))
    scale_4 = np.int64(np.sum(false[scale_4_f]) + np.sum(true[scale_4_t]) + np.ceil(0.4*scale_K))
    scale_5 = np.int64(np.sum(false[scale_5_f]) + np.sum(true[scale_5_t]))
    scale_6 = np.int64(np.sum(false[scale_6_f]) + np.sum(true[scale_6_t]))
    scale_7 = np.int64(np.sum(false[scale_7_f]) + np.sum(true[scale_7_t]) + np.ceil(scale_K))
    scale_8 = np.int64(np.sum(false[scale_8_f]) + np.sum(true[scale_8_t]) + np.ceil(scale_K))
    scale_9 = np.int64(np.sum(false[scale_9_f]) + np.sum(true[scale_9_t]) + np.ceil(0.2*scale_K))
    scale_0 = np.int64(np.sum(false[scale_0_f]) + np.sum(true[scale_0_t]))
    
    sc_lfk = [scale_L,scale_F,scale_K]
    sc_09 = [scale_1,scale_2,scale_3,scale_4,scale_5,scale_6,scale_7,scale_8,scale_9,scale_0]
    
    return np.array(sc_lfk).astype(int), np.array(sc_09).astype(int)
 
def tScore(sex, score_lfk, score_09):
    #loading conversion tables
    df_women = pd.read_excel('conv_W.xlsx')
    df_men = pd.read_excel('conv_M.xlsx')
    if sex == 'female':
        #L
        womenL = df_women.values[:,1]
        womenL = womenL[~np.isnan(womenL)]
        womenL = np.int64(womenL)
        T_scale_L = womenL[score_lfk[0]]

        #F
        womenF = df_women.values[:,2]
        womenF = womenF[~np.isnan(womenF)]
        womenF = np.int64(womenF)
        T_scale_F = womenF[score_lfk[1]]

        #K
        womenK = df_women.values[:,3]
        womenK = womenK[~np.isnan(womenK)]
        womenK = np.int64(womenK)
        T_scale_K = womenK[score_lfk[2]]

        #1
        women1 = df_women.values[:,4]
        women1 = women1[~np.isnan(women1)]
        women1 = np.int64(women1)
        T_scale_1 = women1[score_09[0]]

        #2
        women2 = df_women.values[:,5]
        women2 = women2[~np.isnan(women2)]
        women2 = np.int64(women2)
        T_scale_2 = women2[score_09[1]-8]

        #3
        women3 = df_women.values[:,6]
        women3 = women3[~np.isnan(women3)]
        women3 = np.int64(women3)
        T_scale_3 = women3[score_09[2]-4]

        #4
        women4 = df_women.values[:,7]
        women4 = women4[~np.isnan(women4)]
        women4 = np.int64(women4)
        T_scale_4 = women4[score_09[3]-6]

        #5
        women5 = df_women.values[:,8]
        women5 = women5[~np.isnan(women5)]
        women5 = np.int64(women5)
        T_scale_5 = women5[score_09[4]-15]

        #6
        women6 = df_women.values[:,9]
        women6 = women6[~np.isnan(women6)]
        women6 = np.int64(women6)
        T_scale_6 = women6[score_09[5]]

        #7
        women7 = df_women.values[:,10]
        women7 = women7[~np.isnan(women7)]
        women7 = np.int64(women7)
        T_scale_7 = women7[score_09[6]-7]

        #8
        women8 = df_women.values[:,11]
        women8 = women8[~np.isnan(women8)]
        women8 = np.int64(women8)
        T_scale_8 = women8[score_09[7]-5]

        #9
        women9 = df_women.values[:,12]
        women9 = women9[~np.isnan(women9)]
        women9 = np.int64(women9)
        T_scale_9 = women9[score_09[8]-5]

        #0
        women0 = df_women.values[:,13]
        women0 = women0[~np.isnan(women0)]
        women0 = np.int64(women0)
        T_scale_0 = women0[score_09[9]]
    else:
        #L
        menL = df_men.values[:,1]
        menL = menL[~np.isnan(menL)]
        menL = np.int64(menL)
        T_scale_L = menL[score_lfk[0]]

        #F
        menF = df_men.values[:,2]
        menF = menF[~np.isnan(menF)]
        menF = np.int64(menF)
        T_scale_F = menF[score_lfk[1]]

        #K
        menK = df_men.values[:,3]
        menK = menK[~np.isnan(menK)]
        menK = np.int64(menK)
        T_scale_K = menK[score_lfk[2]]

        #1
        men1 = df_men.values[:,4]
        men1 = men1[~np.isnan(men1)]
        men1 = np.int64(men1)
        T_scale_1 = men1[score_09[0]]

        #2
        men2 = df_men.values[:,5]
        men2 = men2[~np.isnan(men2)]
        men2 = np.int64(men2)
        T_scale_2 = men2[score_09[1]-8]

        #3
        men3 = df_men.values[:,6]
        men3 = men3[~np.isnan(men3)]
        men3 = np.int64(men3)
        T_scale_3 = men3[score_09[2]-8]

        #4
        men4 = df_men.values[:,7]
        men4 = men4[~np.isnan(men4)]
        men4 = np.int64(men4)
        T_scale_4 = men4[score_09[3]-6]

        #5
        men5 = df_men.values[:,8]
        men5 = men5[~np.isnan(men5)]
        men5 = np.int64(men5)
        T_scale_5 = men5[score_09[4]-8]

        #6
        men6 = df_men.values[:,9]
        men6 = men6[~np.isnan(men6)]
        men6 = np.int64(men6)
        T_scale_6 = men6[score_09[5]]

        #7
        men7 = df_men.values[:,10]
        men7 = men7[~np.isnan(men7)]
        men7 = np.int64(men7)
        T_scale_7 = men7[score_09[6]-9]

        #8
        men8 = df_men.values[:,11]
        men8 = men8[~np.isnan(men8)]
        men8 = np.int64(men8)
        T_scale_8 = men8[score_09[7]-7]

        #9
        men9 = df_men.values[:,12]
        men9 = men9[~np.isnan(men9)]
        men9 = np.int64(men9)
        T_scale_9 = men9[score_09[8]-5]

        #0
        men0 = df_men.values[:,13]
        men0 = men0[~np.isnan(men0)]
        men0 = np.int64(men0)
        T_scale_0 = men0[score_09[9]]


    lfk_score = [T_scale_L,T_scale_F,T_scale_K]
    rest_score = [T_scale_1,T_scale_2,T_scale_3,T_scale_4,T_scale_5,T_scale_6,T_scale_7,T_scale_8,T_scale_9,T_scale_0]

    return lfk_score, rest_score

def graphPlotter(sex, name, age, time, lfk, rest):
    fig = plt.figure() # create figure

    ax = fig.add_subplot(1, 1, 1) 

    plt.plot([0,1,2],lfk,label="LFK")
    plt.plot([3,4,5,6,7,8,9,10,11,12],rest,label="0-9")

    plt.scatter([0,1,2],lfk)
    plt.scatter([3,4,5,6,7,8,9,10,11,12],rest)

    ax.set_title ('График MMPI')
    ax.set_ylabel('Т-баллы')
    ax.set_xlabel('Шкала')

    plt.annotate('L', xy=(-0.1, -6))
    plt.annotate(str(lfk[0]), xy=(-0.25, 2))

    plt.annotate('F', xy=(0.9, -6))
    plt.annotate(str(lfk[1]), xy=(0.75, 2))

    plt.annotate('K', xy=(1.9, -6))
    plt.annotate(str(lfk[2]), xy=(1.75, 2))

    plt.annotate('1', xy=(2.9, -6))
    plt.annotate(str(rest[0]), xy=(2.75, 2))

    plt.annotate('2', xy=(3.9, -6))
    plt.annotate(str(rest[1]), xy=(3.75, 2))

    plt.annotate('3', xy=(4.9, -6))
    plt.annotate(str(rest[2]), xy=(4.75, 2))

    plt.annotate('4', xy=(5.9, -6))
    plt.annotate(str(rest[3]), xy=(5.75, 2))

    if sex=='Женский':
        plt.annotate('5Ж', xy=(6.65, -6))
    else:
        plt.annotate('5M', xy=(6.65, -6))
    plt.annotate(str(rest[4]), xy=(6.75, 2))

    plt.annotate('6', xy=(7.9, -6))
    plt.annotate(str(rest[5]), xy=(7.75, 2))

    plt.annotate('7', xy=(8.9, -6))
    plt.annotate(str(rest[6]), xy=(8.75, 2))

    plt.annotate('8', xy=(9.9, -6))
    plt.annotate(str(rest[7]), xy=(9.75, 2))

    plt.annotate('9', xy=(10.9, -6))
    plt.annotate(str(rest[8]), xy=(10.75, 2))

    plt.annotate('0', xy=(11.9, -6))
    plt.annotate(str(rest[9]), xy=(11.75, 2))

    srt_arr = np.zeros((10,2))
    rest_lab = [1,2,3,4,5,6,7,8,9,0]

    #display order of scales in descent
    for i in range(10):
        srt_arr[i,0] = rest_lab[i]
        srt_arr[i,1] = rest[i]

    srt_arr=sorted(srt_arr, key=lambda label:label[1], reverse=True)
    print_order = []
    for i in range(10):
        print_order.append(str(np.int64(srt_arr[i][0])))

    print_order= ''.join(print_order)
    plt.annotate(print_order, xy=(-0.1, -15))

    plt.grid(True)


    plt.axhline(y=np.min(rest)+(np.max(rest)-np.min(rest))/2, xmin=0.25, xmax=1, color='#d62728', alpha=0.4, label="изолиния")
    # Draw a default hline at y=.5 that spans the middle half of the axes
    plt.axhline(y=-15, xmin=0, xmax=1,alpha=0.01)
    plt.axhline(y=120, xmin=0, xmax=1,alpha=0.01)
    plt.axhline(y=70, xmin=0, xmax=1,alpha=0.35,color='#00ff00')
    plt.axhline(y=30, xmin=0, xmax=1,alpha=0.35,color='#00ff00')
    plt.legend(loc='upper left')
    plotLocation = str(name)+'_'+str(age)+'_'+str(time)+'_'+'mmpi_plt.png'
    
    try:
        os.mkdir("profiles/"+str(name)+"/")
        plt.savefig("profiles/"+str(name)+"/"+plotLocation)
    except:
        plt.savefig("profiles/"+str(name)+"/"+plotLocation)

    print(plotLocation)
    return plotLocation
    
class Person:
    """
    Simple class for representing a point in a Cartesian coordinate system.
    """
    
    def __init__(self, firstName, lastName, sex, age, work, birthDate, notes, imageLocation, modelLocation):
        self.firstName = firstName
        self.lastName = lastName
        if sex == 'male':
            self.sex = 'Мужской'
        else: 
            self.sex = 'Женский'

        
        self.age = age
        self.work = work
        self.testDate = datetime.now().strftime('%d_%m_%Y')
        self.birthDate = birthDate
        self.notes = notes
        self.plotLoc = ''
        
        image = cv2.imread(imageLocation)
        model = loadModel(modelLocation)
        warped = formTransformation(image)
        answer_locs = ROIextractor(model, warped)
        self.tr, self.fl =  locationSorting(answer_locs)
        score_lfk, score_09 = rawScoreCounter(self.tr, self.fl)
        
        self.lfk, self.rest = tScore(sex, score_lfk, score_09)
        
    def plot(self):
        """
        plot the graph
        """
        return graphPlotter(self.sex, str(self.firstName + self.lastName), self.age, datetime.now().strftime('%d%m%Y_%H%M%S'), self.lfk, self.rest)
        
        
    def demographics(self):
        print(self.firstName + ' ' + self.lastName)
        print('Пол: ' + self.sex)
        print('Возраст: ' + self.age)
        print('Сфера дефтельности: ' + self.work)
        print('Дата теста: ' + self.testDate)
        print('Дата рождения: ' + self.birthDate)
        
    def writeToTextFile(self):
        with open("profiles/"+self.firstName+self.lastName+"/"+self.firstName+"_"+self.lastName+"_"+datetime.now().strftime('%d%m%Y_%H%M%S') + ".txt", 'w') as file:
            file.write("Имя: " + self.firstName + "\n" +
                       "Фамилия: " + self.lastName + "\n" +
                       "Пол: " + self.sex + "\n" + 
                       "Возраст: " + self.age + "\n" + 
                       "Сфера деятельности: " + self.work + "\n" + 
                       "Дата теста: " + self.testDate + "\n" + 
                       "Дата рождения: " + self.birthDate + "\n" + 
                       "Заметки: " + str(self.notes) + "\n" + 
                       "LFK: " + str(self.lfk) + "\n" + 
                       "Шкалы: " + str(self.rest) + "\n")

        with open("profiles/"+self.firstName+self.lastName+"/"+self.firstName+"_"+self.lastName+"_"+datetime.now().strftime('%d%m%Y_%H%M%S') + "true.txt", 'w') as file:
                    file.write(str(self.tr))

        with open("profiles/"+self.firstName+self.lastName+"/"+self.firstName+"_"+self.lastName+"_"+datetime.now().strftime('%d%m%Y_%H%M%S') + "false.txt", 'w') as file:
                    file.write(str(self.fl))
