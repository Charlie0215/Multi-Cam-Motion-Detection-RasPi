import cv2
import numpy as np
import random
import argparse
import os

from nms import non_max_suppression_fast

imagePath = '/Users/dai/Desktop/Project/Multi-Cam-Motion-Detection-RasPi/pedestrian-detection/images/person_010.bmp'

# construct the argument parse and parse the arguments
#parser = argparse.ArgumentParser()
#parser.add_argument('-i', dest='images', type=str, required=True, help='path to image directory')
#opt = vars(parser.parse_args())

def predestrain_detection(image):

	# initialize the HOG descriptor/person detector
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	width = min(400, image.shape[0])
	height = min(400, image.shape[1])
	image = cv2.resize(image, (height, width))
	orig = image.copy()
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
	# draw the original bounding box
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)


	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlay theshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression_fast(rects, overlapThresh = 0.65)


	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255,0), 2)
	print("[INFO] {}: {} original boxes, {} after suppression".format(
		'120', len(rects), len(pick)))
	cv2.imshow('original', orig)
	cv2.imshow('after nms', image)
	cv2.waitKey(3000)

if __name__ == '__main__':
	image = cv2.imread(imagePath)
	predestrain_detection(image)