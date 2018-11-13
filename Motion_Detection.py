import cv2
import numpy as np
import datetime
import time

class BasicMotionDetector:
	def __init__(self, accumWeight=0.5, deltaThresh=10, minArea=3000, maxArea=6000):

		self.accumWeight = accumWeight
		self.deltaThresh = deltaThresh
		self.minArea = minArea
		self.maxArea = maxArea
		self.avg = None

	def update(self, image):
		locs = []
	
		if self.avg is None:
			self.avg = image.astype("float")
			return locs

		# accumulate the weighted average between
		# the current frame and the previous frames
		cv2.accumulateWeighted(image, self.avg, self.accumWeight)
		frameDelta = cv2.absdiff(image, cv2.convertScaleAbs(self.avg))

		# threshold the delta image and apply a series of dilations
		# to help fill in holes
		ret, thresh = cv2.threshold(frameDelta, self.deltaThresh, 255,
			cv2.THRESH_BINARY)
		thresh = cv2.dilate(thresh, None, iterations=2)

		im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		for c in contours:

			if cv2.contourArea(c) > self.minArea and cv2.contourArea(c) < self.maxArea:
				locs.append(c)

		# locations
		return locs

def initialization():
	print("[INFO] starting cameras...")
	time.sleep(2.0)


def main():
	# initialize the camera streams
	cap1 = cv2.VideoCapture(0)
	cap2 = cv2.VideoCapture(1)
	
	cam1Motion = BasicMotionDetector()
	cam2Motion = BasicMotionDetector()
	
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
				total +=1
				# change image to gray scale for motion detector
				# and apply gaussian bluring
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				gray = cv2.GaussianBlur(gray, (21, 21), 0)
				
				locs = motion.update(gray)
				print(total)
				if total < 8:
					frames.append(frame)
					continue
				if total > 64:
					total = 0
				if len(locs) > 0:
					
					# initialize minimum and maximum coordinates
					(minX, minY) = (np.inf, np.inf)
					(maxX, maxY) = (-np.inf, -np.inf)
					
					for l in locs:
						(x, y, w, h) = cv2.boundingRect(l)
						(minX, maxX) = (min(minX, x), max(maxX, x + w))
						(minY, maxY) = (min(minY, y), max(maxY, y + h))	
					
					# draw the bounding box
					cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 3)
				
				# update the frames list
				frames.append(frame)
				
			
			# loop over the frames a second time
			for (frame, name) in zip(frames, ("cam0", "cam1")):
				
				cv2.imshow(name, frame)
					
		else:
			break
		# the program will stop if you press 'q' key
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == '__main__':
	
	initialization()
	main()
	cv2.destroyAllWindows()
