import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def thresholding():
    image_path = r"C:\Users\karen\OneDrive\Escritorio\DIP\250206\1-037.JPG"
    img = cv.imread(image_path)
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    hist = cv.calcHist([imgGray],[0],None,[256],[0,256])
    plt.figure()
    plt.plot(hist)
    plt.xlabel('bins')
    plt.ylabel('# pixels')
    plt.show()

    thresOptions =[cv.THRESH_BINARY,
                   cv.THRESH_BINARY_INV,
                   cv.THRESH_TOZERO,
                   cv.THRESH_TOZERO_INV,
                   cv.THRESH_BINARY + cv.THRESH_OTSU]
    thresNames = ['Binary', 'Binary Inverted','toZero', 'toZero Inverted', 'Otsu']

    plt.figure()
    plt.subplot(231)
    plt.imshow(imgGray, cmap='gray')
    

    for i in range(len(thresOptions)):
        plt.subplot(2,3,i+2)
        _, imgThres = cv.threshold(imgGray,100,255,thresOptions[i])
        plt.imshow(imgThres, cmap='gray')
        plt.title(thresNames[i])

    plt.show()

if __name__ == '__main__':
    thresholding()
