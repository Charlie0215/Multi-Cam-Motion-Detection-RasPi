import cv2
import numpy as np
import random
import argparse
import os
import math
from utils import select_ROI

import threading

lock1 = threading.Lock()

kernel_size = 11
imagePath = '/Users/dai/Desktop/Project/Multi-Cam-Motion-Detection-RasPi/pedestrian-detection/images/person_010.bmp'
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# construct the argument parse and parse the arguments
#parser = argparse.ArgumentParser()
#parser.add_argument('-i', dest='images', type=str, required=True, help='path to image directory')
#opt = vars(parser.parse_args())
def haar_cascade_setection(image, classifier):

	
	sizes = []

	#height = min(700, image.shape[0])
	#width = min(500, image.shape[1])
	height = min(350, image.shape[0])
	width = min(250, image.shape[1])
	#height = math.ceil(image.shape[0] / 2)
	#width = math.ceil(image.shape[1] / 2)
	image = cv2.resize(image, (height, width))
	image_size = float(height * width)
	print("#"*30, image.shape)
	img = image.copy()
	img = select_ROI(img)
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	masked_image = select_ROI(image)
	#image = cv2.GaussianBlur(image,(kernel_size,kernel_size),0)
	image = cv2.blur(image, (kernel_size, kernel_size))
	#img = cv2.medianBlur(img,kernel_size)
	people = classifier.detectMultiScale(masked_image, 1.2, 3)
	'''
	for (x,y,w,h) in people:
		size = (w * h).astype('float')
		if size / image_size < 0.05:
			pass 
		if size / image_size > 0.25:
			sizes.append(size)
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	
	'''
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in people])
	pick = non_max_suppression_fast(rects, overlapThresh = 0.6)
	for (xA, yA, xB, yB) in pick:
		size = (xB - xA) * (yB - yA).astype('float')
		if size / image_size < 0.03:
			#image = cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255,0), 2)
			pass
		else:
			image = cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255,0), 2)

	for rect in rects:
		size = (rect[2] - rect[0]) * (rect[3] - rect[1]).astype('float') 
		if size / image_size > 0.11 and size / image_size < 0.2:
			sizes.append(size)
	

	return image, sizes


def predestrain_detection(image, hog):

	# initialize the HOG descriptor/person detector
	sizes = []

	width = min(500, image.shape[0])
	height = min(500, image.shape[1])
	#width = image.shape[0]
	#height = image.shape[1]
	image_size = float(height * width)
	image = cv2.resize(image, (height, width))
	orig = image.copy()
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
	# draw the original bounding box
	#for (x, y, w, h) in rects:
	#	cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)


	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlay theshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression_fast(rects, overlapThresh = 0.6)


	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		size = (xB - xA) * (yB - yA).astype('float')
		if size / image_size < 0.05:
			pass 
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255,0), 2)

	for rect in rects:
		size = (rect[2] - rect[0]) * (rect[3] - rect[1]).astype('float') 
		if size / image_size > 0.3:
			sizes.append(size)

	return image, sizes

def non_max_suppression_fast(boxes, overlapThresh):
	# if there no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == 'i':
		boxes = boxes.astype('float')
	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and sort the bounding boxes
	# by the bottom-right y-coordinate of the bounding box

	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looking while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x,y) coordinates for the start of
		# the bounding box and the smallest (x,y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the weight and height of the bounding box
		w = np.maximum(0, xx1 - xx2 + 1)
		h = np.maximum(0, yy1 - yy2 + 1)

		# compute the ratio of the bounding box
		overlap = (w*h) / area[idxs[:last]]

		# delete all indexes from the index list that have 
		idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype('int')


if __name__ == '__main__':
	image = cv2.imread(imagePath)
	orig, sizes = predestrain_detection(image, hog)
	cv2.imshow('image', orig)
	image_size = image.shape[0] * image.shape[1]
	print(sizes, image_size)
