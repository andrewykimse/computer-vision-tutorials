import cv2 as cv
import time
import numpy as np

## Preparation for writing the ouput video
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# reading from the webcam
cap = cv.VideoCapture(0)

# allow the system to sleep for 3 seconds before the webcam starts
time.sleep(3)
count = 0
background = 0

# capture the background in range of 60
for i in range(60):
	ret, background = cap.read()
background = np.flip(background, axis=1)

# Read every frame from the webcam until the camera is open
while (cap.isOpened()):
	ret, img = cap.read()
	if not ret:
		break
	count += 1
	img = np.flip(img, axis=1)
	
	# convert the color space from BGR to HSV
	hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	
	# generate masks to detect red color
	lower_red = np.array([0, 125, 50])
	upper_red = np.array([10, 255, 255])
	mask1 = cv.inRange(hsv, lower_red, upper_red)

	lower_red = np.array([170, 120, 70])
	upper_red = np.array([180, 255, 255])
	mask2 = cv.inRange(hsv, lower_red, upper_red)

	mask1 = mask1 + mask2

	# Open and Dilate the mask image
	mask1 = cv.morphologyEx(mask1, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
	mask1 = cv.morphologyEx(mask1, cv.MORPH_DILATE, np.ones((3, 3), np.uint8))

	# Create an inverted mask to segment out the red color from the frame
	mask2 = cv.bitwise_not(mask1)

	# Segment the red color part out of the frame using bitwise and with the inverted mask
	res1 = cv.bitwise_and(img, img, mask=mask2)

	# Create the image showing static background frame pixels only for the masked region
	res2 = cv.bitwise_and(background, background, mask=mask1)

	# Generate the final output and writing
	finalOutput = cv.addWeighted(res1, 1, res2, 1, 0)
	out.write(finalOutput)
	cv.imshow("magic", finalOutput)
	cv.waitKey(1)

cap.release()
out.release()
cv.destroyAllWindows()
