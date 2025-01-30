import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import sklearn
import jupyter
import cv2

# Show image
image = cv2.imread(r'C:\Users\karen\OneDrive\Escritorio\DIP\250123\brain.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('Image', image)
cv2.waitKey(0)

#Addition (brighter)
cte_add = 100
imageAdd = cv2.add(image, cte_add)
cv2.imshow('Image Add', imageAdd)
cv2.waitKey(0)

#Substraction 
imageSub = cv2.subtract(image, cte_add)
cv2.imshow('Image Sub', imageSub)
cv2.waitKey(0)

#Product (increase contrast, expand hist) cte entre 0 y 2 para no entrar en saturacion, RGB apply to each channel
cte_pro = 1.8
imagePro = cv2.multiply(image, cte_pro)
cv2.imshow('Image prod', imagePro)
cv2.waitKey(0)

#Division
cte_div = 0.8
imageDiv = cv2.multiply(image, cte_div)
cv2.imshow('Image div', imageDiv)
cv2.waitKey(0)

# Histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
plt.bar(range(256), hist[:, 0], width=1, color='gray', alpha=0.7)
plt.title('Histogram')
plt.xlabel('Intensity')
plt.ylabel('Pixels')
plt.show()

cv2.destroyAllWindows()