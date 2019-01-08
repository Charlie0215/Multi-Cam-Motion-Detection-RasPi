import cv2
import numpy as np
import datetime
import time
import os
import math


from gpio import *
#from Motion_Detection import BasicMotionDetector
from predestrain_detect import predestrain_detection, haar_cascade_setection

#camera1 = camera()
#camera2 = camera()


def cams_2():
	# initialize hog descriptor
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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
			frame1 = cv2.resize(frame1, (400, 400))
			frame2 = cv2.resize(frame2, (400, 400))
			
			# flip the images if you want it shows properly on your screen
			frame1 = cv2.flip(frame1, 1)
			frame2 = cv2.flip(frame2, 1)

			for (frame, motion) in zip((frame1, frame2), (cam1Motion, cam2Motion)):
				frame, size = predestrain_detect(frame, hog)
				frames.append(frame)

			# loop over the frames a second time
			for (frame, name) in zip(frames, ("cam0", "cam1")):
				
				cv2.imshow(name, frame)
					
		else:
			break
		# the program will stop if you press 'q' key
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

def cam_1():
	# initialize hog descriptor
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# initialize the camera streams
	cap1 = cv2.VideoCapture(1)
	
	# Check if cameras are connected 
	if cap1.isOpened == False:
		print('Cannot open camera1')
	
	total = 0
	# loop through the camera stream
	while True:
		frames = []
		
		flag1, frame1 = cap1.read()
		
		# check frame availability again
		if flag1 == True:
			frame1 = cv2.resize(frame1, (400, 400))
			image_size = frame.shape[0] * frame.shape[1]
			# flip the images if you want it shows properly on your screen
			frame1 = cv2.flip(frame1, 1)
			frame1, sizes = predestrain_detect(frame, hog)

			cv2.imshow('cam', frame1)
			for size in sizes():
				print("danger!!!")
					
		else:
			break
		# the program will stop if you press 'q' key
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
def test_image(path):
	files = [f for f in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, f))]
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	for file in files:
		file_path = os.path.join(path, file)
		img = cv2.imread(file_path)
		w = math.ceil(img.shape[0] / 2)
		h = math.ceil(img.shape[1] / 2)
		img, sizes = predestrain_detection(img, hog)
		img = cv2.resize(img, (h, w))
		cv2.imshow('image', img)
		if len(sizes) > 0:
			for size in sizes:
				print("dangers!!!")
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

def test_video(path):
	#camera1 = camera(1)
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	body_classifier = cv2.CascadeClassifier('haarcascade_pedestrian.xml')
	cap = cv2.VideoCapture(path)
	threads = []
	if cap.isOpened() == False:
		print("cannot open video file")
	index = 0
	while True:
		flag, img = cap.read()
		#camera2.off()
		
		if flag == True and index == 3:
			index = 0
			w = math.ceil(img.shape[0] / 2)
			h = math.ceil(img.shape[1] / 2)
			#img, sizes = predestrain_detection(img, hog)
			img, sizes = haar_cascade_setection(img, body_classifier)
			img = cv2.resize(img, (h, w))
			
			if len(sizes) > 0:
				print('danger!')
				text = 'dangerous!'
				
				cv2.putText(img, text, (50, 50),
				 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0),
				  lineType=cv2.LINE_AA)
				#camera1.thread()
				#thread = camera(1)
				#thread.start()
				#camera1.off()
				#camera1.stop()
				thread = vibrate_thread(1, "vibrate", 1, 1)	
				thread.start()
				#thread.join()
			cv2.imshow('image', img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				
				break
		else:
			index += 1
			print(index)
	
	cap.release()
	cv2.destroyAllWindows()



if __name__ == '__main__':
	#test('/Users/dai/Downloads/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_00')
	test_video('LONDONWALKOxfordStreettoCarnabyStreetEngland.mp4')



