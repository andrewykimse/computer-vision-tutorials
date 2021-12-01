import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Open the Image
img = cv.imread('images/sonic.png',0)
# Apple Canny Edge Detection
edges = cv.Canny(img, 100, 200)

plt.subplot(121), plt.imshow(img, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
