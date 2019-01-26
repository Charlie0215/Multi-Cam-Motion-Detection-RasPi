import datetime
import threading
from threading import Thread
import cv2
import argparse
import time
from predestrain_detect import *

lock = threading.Lock()


class FPS:
	def __init__(self):
		# show the start time, end time, and total number of frames
		# that are examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		#print('*'*30, self._start)
		return self

	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
		#print('*'*30, self._end)
		return self

	def update(self):
		# incremental the number of frames examined during
		# the start and end intervals
		self._numFrames += 1

	def elapse(self):
		# retuen the total number of time between the 
		# start and end intervals
		return (self._end - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapse()


class WebcamVideoStream:
	def __init__(self, input):#, src):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(input)
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the variable used to indicate if the thread should 
		# be stopped
		self.stopped = False
		#self.src = src

	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping inifiniteky until the thread is stopped
		while True:
			# if the thread indicator variable is setm stop the thread
			if self.stopped:
				return
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read

		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

class WebcamVideoStream_main_thread:
	def __init__(self, video_path, src, model_path):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(video_path)
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the variable used to indicate if the thread should 
		# be stopped
		self.stopped = False
		self.src = src
		self.body_classifier = cv2.CascadeClassifier(model_path)

	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping inifiniteky until the thread is stopped
		while True:
			# if the thread indicator variable is setm stop the thread
			if self.stopped:
				return
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		return self.frame
		'''
		# return the frame most recently read
		print("processing cam ", self.src)
		img = self.frame
		print("*"*30, img.shape)
		w = math.ceil(img.shape[0] / 2)
		h = math.ceil(img.shape[1] / 2)
		image = cv2.resize(img, (h, w))
		image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#cv2.imshow('cam1', image)
		
		while True:
			#img, sizes = predestrain_detection(img, hog)
			lock.acquire()
			img, sizes = haar_cascade_setection(img, self.body_classifier)
			lock.release()
			img = cv2.resize(img, (h, w))
			
			if len(sizes) > 0:
				print('danger on', self.src)
				text = 'dangerous!'
				
				cv2.putText(img, text, (50, 50),
				 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0),
				  lineType=cv2.LINE_AA)
		

		return img
		'''
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

if __name__ == '__main__':
	opt = argparse.ArgumentParser()
	opt.add_argument('-d', "--display", default=0, type=int, help='1 to display frames')
	args = vars(opt.parse_args())

	vs = WebcamVideoStream(0).start()
	time.sleep(0.2)
	fps = FPS().start()

	while fps._numFrames < 4000:
		frame = vs.read()
		frame = cv2.resize(frame, (400, 400))
		if args['display'] == 1:
			cv2.imshow('image', frame)
			key = cv2.waitKey(1) & 0xFF
		fps.update()

	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapse()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	cv2.destroyAllWindows()
	vs.stop()