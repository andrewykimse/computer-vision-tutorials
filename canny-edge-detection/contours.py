import cv2 as cv
import numpy as np

image = cv.imread('images/contours.jpg')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# apply binary thresholding
ret, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)


# visualize the binary image
#cv.imshow('Binary image', thresh)
#cv.waitKey(0)
#cv.imwrite('image_thres1.jpg', thresh)
#cv.destroyAllWindows()


# CHAIN_APPROX_NONE METHOD
#contours, hieracrchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

# draw contours on the original image
#image_copy = image.copy()
#cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

# see the results
#cv.imshow('None approximation', image_copy)
#cv.waitKey(0)
#cv.imwrite('contours_none_contours.jpg', image_copy)
#cv.destroyAllWindows()




# CHAIN_APPROX_SIMPLE METHOD
contours1, hierarchy1 = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
image_copy1 = image.copy()
cv.drawContours(image_copy1, contours1, -1, (0,255,0), 2, cv.LINE_AA)
cv.imshow('Simple approximation', image_copy1)
cv.waitKey(0)
cv.imwrite('contours_simple_contours1.jpg', image_copy1)
cv.destroyAllWindows()







