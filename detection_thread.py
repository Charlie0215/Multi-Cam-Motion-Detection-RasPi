import threading
import cv2
import numpy as np
import math
from predestrain_detect import predestrain_detection, haar_cascade_setection
from detection_thread import *

lock = threading.Lock()

class cam_detection(threading.Thread):
	def __init__(self,  cam_id, model_path, video_path):
		threading.Thread.__init__(self)
		self.hog = cv2.HOGDescriptor()
		self.body_classifier = cv2.CascadeClassifier(model_path)
		#self.cap = cv2.VideoCapture(video_path)
		self.cam_id = cam_id
		self.path = video_path
	def run(self):
		global n, lock
		lock.acquire()
		test_video(self.path)
		lock.release()
'''
def detection(cam_id):
	self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	if self.cap.isOpened() == False:
		print("cannot open video file")
	index = 0
	while True:
		flag, img = self.cap.read()
		#camera2.off()
		w = math.ceil(img.shape[0] / 2)
		h = math.ceil(img.shape[1] / 2)
		image = cv2.resize(img, (h, w))
		image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		window_name = 'image' + str(cam_id)
		cv2.imshow(window_name, image)
		if flag == True and index == 3:
			index = 0
			#img, sizes = predestrain_detection(img, hog)
			img, sizes = haar_cascade_setection(img, self.body_classifier)
			img = cv2.resize(img, (h, w))
			
			if len(sizes) > 0:
				print('danger!')
				text = 'dangerous!'
				
				cv2.putText(img, text, (50, 50),
				 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0),
				  lineType=cv2.LINE_AA)
			window_name = 'image' + str(cam_id)
			cv2.imshow("window_name", img)
			#print("show image")
			if cv2.waitKey(1) & 0xFF == ord('q'):
				
				break
		else:
			index += 1
			print(index)

	cap.release()
	cv2.destroyAllWindows()
'''
def test_video(path):
	#camera1 = camera(1)
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	body_classifier = cv2.CascadeClassifier('haarcascade_pedestrian.xml')
	cap1 = cv2.VideoCapture(path)
	threads = []
	if cap1.isOpened() == False:
		print("cannot open video file")
	index = 0
	while True:
		flag, img = cap1.read()
		#camera2.off()
		w = math.ceil(img.shape[0] / 2)
		h = math.ceil(img.shape[1] / 2)
		image = cv2.resize(img, (h, w))
		image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cv2.imshow('image', image)
		if flag == True and index == 3:
			index = 0
			#img, sizes = predestrain_detection(img, hog)
			img, sizes = haar_cascade_setection(img, body_classifier)
			img = cv2.resize(img, (h, w))
			
			if len(sizes) > 0:
				print('danger!')
				text = 'dangerous!'
				
				cv2.putText(img, text, (50, 50),
				 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0),
				  lineType=cv2.LINE_AA)

			cv2.imshow('image', img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				
				break
		else:
			index += 1
			print(index)
	
	cap.release()
	cv2.destroyAllWindows()