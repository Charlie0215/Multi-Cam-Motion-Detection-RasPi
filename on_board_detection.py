import cv2
import numpy as np
import datetime
import time

from gpio import *
from Motion_Detection import BasicMotionDetector
from predestrain_detect import predestrain_detect


def main():
	# initialize the camera streams
	cap1 = cv2.VideoCapture(0)
	cap2 = cv2.VideoCapture(1)
	
	# Check if cameras are connected 
	if cap1.isOpened == False:
		print('Cannot open camera1')
	if cap2.isOpened == False:
		print('Cannot open camera2')
	
	total = 0
	# loop through the camera stream
	while True:
		frames = []
		
		flag1, frame1 = cap1.read()
		flag2, frame2 = cap2.read()
		
		# check frame availability again
		if flag1 == True and flag2 == True: 
			'''
			implementation of object detection function HERE
			'''

			# loop over the frames a second time
			for (frame, name) in zip(frames, ("cam0", "cam1")):
				
				cv2.imshow(name, frame)
					
		else:
			break
		# the program will stop if you press 'q' key
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


if __name__ == '__main__':
	main()

