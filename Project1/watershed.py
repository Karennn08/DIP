import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def watershed():
    image_path = r"C:\Users\karen\OneDrive\Escritorio\DIP\250206\1-037.JPG"
    img = cv.imread(image_path)
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap='gray')

    plt.subplot(232)
    _, imgThreshold = cv.threshold(img,100,255,cv.THRESH_BINARY_INV)
    plt.imshow(imgThreshold, cmap='gray')

    plt.subplot(233)
    kernel = np.ones((3,3),np.uint8)
    imgDilate = cv.morphologyEx(imgThreshold, cv.MORPH_DILATE, kernel)
    plt.imshow(imgDilate)

    plt.subplot(234)
    distTrans = cv.distanceTransform(imgDilate,cv.DIST_L2,5)
    plt.imshow(distTrans)

    plt.subplot(235)
    _, distThres = cv.threshold(distTrans, 15 ,255, cv.THRESH_BINARY)
    plt.imshow(distThres)

    plt.subplot(236)
    distThres = np.uint8(distThres)
    _,labels = cv.connectedComponents(distThres)
    plt.imshow(labels)

    plt.figure()
    plt.subplot(121)
    labels = np.int32(labels)
    labels = cv.watershed(imgRGB,labels)
    plt.imshow(labels)

    plt.subplot(122)
    imgRGB[labels ==-1] = [255,0,0]
    plt.imshow(imgRGB)
    
    plt.show()


if __name__ == '__main__':
    watershed()


