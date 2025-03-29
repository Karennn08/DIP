import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def otsuThreshold():
    image_path = r"C:\Users\karen\OneDrive\Escritorio\DIP\250206\1-037.JPG"
    img = cv.imread(image_path)
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(131)
    plt.imshow(imgGray, cmap='gray')
    plt.title('Gray')

    plt.subplot(132)
    _, imgThres = cv.threshold(imgGray,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
    plt.imshow(imgThres, cmap='gray')
    plt.title('Otsu')

    plt.show()


if __name__ == '__main__':
    otsuThreshold()
