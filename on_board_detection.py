import cv2
import numpy as np
import datetime
import time
import os
import math
import argparse
import threading

#from gpio import *
#from Motion_Detection import BasicMotionDetector
from predestrain_detect import predestrain_detection, haar_cascade_setection
from detection_thread import *
from thread_io import *
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

def test_video_pair(path1, path2):
	#camera1 = camera(1)
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	body_classifier = cv2.CascadeClassifier('haarcascade_pedestrian.xml')
	cap1 = cv2.VideoCapture(path1)
	cap2 = cv2.VideoCapture(path2)
	threads = []
	if cap1.isOpened() == False:
		print("cannot open video file")
	if cap2.isOpened() == False:
		print("cannot open video file")
	index = 0
	while True:
		flag1, img1 = cap1.read()
		flag2, img2 = cap1.read()
		#camera2.off()
		w = math.ceil(img1.shape[0] / 2)
		h = math.ceil(img1.shape[1] / 2)
		image1 = cv2.resize(img1, (h, w))
		image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		image2 = cv2.resize(img2, (h, w))
		image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		cv2.imshow('cam1', image1)
		cv2.imshow('cam2', image2)
		#if flag1 == True and index == 3:
		if index == 3:
			index = 0
			#img, sizes = predestrain_detection(img, hog)
			img1, sizes1 = haar_cascade_setection(img1, body_classifier)
			img2, sizes2 = haar_cascade_setection(img2, body_classifier)

			img1 = cv2.resize(img1, (h, w))
			img2 = cv2.resize(img1, (h, w))
			
			if len(sizes1) > 0:
				print('danger on 1!')
				text = 'dangerous!'
				
				cv2.putText(img1, text, (50, 50),
				 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0),
				  lineType=cv2.LINE_AA)

			if len(sizes2) > 0:
				print('danger on 2!')
				text = 'dangerous!'
				
				cv2.putText(img2, text, (50, 50),
				 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0),
				  lineType=cv2.LINE_AA)

			cv2.imshow('cam1', img1)
			cv2.imshow('cam2', img2)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				
				break
		else:
			index += 1
			print(index)
	
	cap.release()
	cv2.destroyAllWindows()


# Threading class implementation
def test_video_pair_2(path1, path2):
	thread1 = cam_detection(1, 'haarcascade_pedestrian.xml', path1)
	#thread2 = cam_detection(2, 'haarcascade_pedestrian.xml', path2)
	thread1.start()
	#thread2.start()

def test_video_pair_3(path1, path2):
	opt = argparse.ArgumentParser()
	opt.add_argument('-d', '--d', default=1, type=int, help='1: display image')
	arg = vars(opt.parse_args())
	vs1 = WebcamVideoStream(path1, 1).start()
	vs2 = WebcamVideoStream(path2, 2).start()
	body_classifier = cv2.CascadeClassifier('haarcascade_pedestrian.xml')
	fps = FPS().start()
	index = 0
	for i in range(40):

		img1 = vs1.read()
		img2 = vs2.read()
		w = math.ceil(img1.shape[0] / 2)
		h = math.ceil(img1.shape[1] / 2)
		image1 = cv2.resize(img1, (h, w))
		image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		image2 = cv2.resize(img2, (h, w))
		image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		cv2.imshow('cam1', image1)
		cv2.imshow('cam2', image2)
		if index == 3:
			index = 0
			#img, sizes = predestrain_detection(img, hog)
			img1, sizes1 = haar_cascade_setection(img1, body_classifier)
			img2, sizes2 = haar_cascade_setection(img2, body_classifier)

			img1 = cv2.resize(img1, (h, w))
			img2 = cv2.resize(img1, (h, w))
			
			if len(sizes1) > 0:
				print('danger on 1!')
				text = 'dangerous!'
				
				cv2.putText(img1, text, (50, 50),
				 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0),
				  lineType=cv2.LINE_AA)

			if len(sizes2) > 0:
				print('danger on 2!')
				text = 'dangerous!'
				
				cv2.putText(img2, text, (50, 50),
				 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0),
				  lineType=cv2.LINE_AA)
			if arg['d'] == 1:
				Thread(target=show_image, args=(img1)).start()
				Thread(target=show_image, args=(img2)).start()
				#cv2.imshow('cam2', img2)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				
				break
		else:
			index += 1
			print(index)
		fps.update()

	print("[INFO] elapsed time: {:.2f}".format(fps.elapse()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	fps.stop()	
	cap.release()
	cv2.destroyAllWindows()

def show_image(img):
	cv2.imshow("image", img)
	return


def test_video_pair_4(path1, path2):
	opt = argparse.ArgumentParser()
	opt.add_argument('-d', '--d', default=1, type=int, help='1: display image')
	vs1 = WebcamVideoStream_main_thread(path1, 1, 'haarcascade_pedestrian.xml').start()
	vs2 = WebcamVideoStream_main_thread(path2, 2, 'haarcascade_pedestrian.xml').start()
	fps = FPS().start()
	index = 0
	#while True:
	for i in range(40):
		img1 = vs1.read()
		img2 = vs2.read()
		if index == 3: 
			cv2.imshow('cam1', img1)
			cv2.imshow('cam2', img2)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			index += 1
			print(index)
		fps.update()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapse()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	fps.stop()
	vs.stop()
	cv2.distroyAllWindows()
	


if __name__ == '__main__':
	#test('/Users/dai/Downloads/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_00')
	#test_video_pair_2('LONDONWALKOxfordStreettoCarnabyStreetEngland.mp4', 'LONDONWALKOxfordStreettoCarnabyStreetEngland2.mp4')
	test_video_pair_4('LONDONWALKOxfordStreettoCarnabyStreetEngland.mp4', 'LONDONWALKOxfordStreettoCarnabyStreetEngland2.mp4')
	#test_video('LONDONWALKOxfordStreettoCarnabyStreetEngland.mp4')



