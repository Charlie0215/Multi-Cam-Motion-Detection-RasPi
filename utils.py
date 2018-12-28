import cv2
import numpy as np
import os

def cvt_frame_to_video(pathIn,pathOut,fps):
	frame = []
	files = [f for f in sorted(os.listdir(pathIn)) if os.path.isfile(os.path.join(pathIn, f))]
	
	for i in range(len(files)):
		image_path = os.path.join(pathIn, files[i])
		img = cv2.imread(image_path)
		h, w, d = img.shape
		size = (w, h)
		print(image_path)
		# inserting the frame into an image array
		frame.append(img)
	frame_width = frame[0].shape[0]
	frame_height = frame[1].shape[1]
	fourcc = cv2.VideoWriter_fourcc(*'X264')
	out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

	for i in range(len(frame)):
		print(i)
		#print(frame)
		out.write(frame[i])
	out.release()

def read_video():
	cap = cv2.VideoCapture('output.mp4')

	while(cap.isOpened()):
		ret, frame = cap.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		cv2.imshow('frame',gray)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
'''
def main():
	import cv2
	import numpy as np

	# Create a VideoCapture object
	cap = cv2.VideoCapture(0)

	# Check if camera opened successfully
	if (cap.isOpened() == False): 
		print("Unable to read camera feed")

	# Default resolutions of the frame are obtained.The default resolutions are system dependent.
	# We convert the resolutions from float to integer.
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))

	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

	while(True):
		ret, frame = cap.read()

	if ret == True: 

		# Write the frame into the file 'output.avi'
		out.write(frame)

		# Display the resulting frame    
		cv2.imshow('frame',frame)

		# Press Q on keyboard to stop recording
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Break the loop
	else:
		break 

	# When everything done, release the video capture and video write objects
	cap.release()
	out.release()

	# Closes all the frames
	cv2.destroyAllWindows() 
'''
def select_ROI(image):
	# mask defaulting to black for 3-channel and transparent for 4-channel
	mask = np.zeros(image.shape, np.uint8)

	lower_left_x = 0 
	lower_left_y = 0.9 * image.shape[0]
	upper_left_x = 0.1 * image.shape[1]
	upper_left_y = 0.35 * image.shape[0]
	upper_right_x = 0.9 * image.shape[1]
	upper_right_y = 0.35 * image.shape[0]
	lower_right_x = image.shape[1]
	lower_right_y = 0.9 * image.shape[0]

	roi_corners = np.array([[[lower_left_x, lower_left_y],[upper_left_x, upper_left_y],
	 [upper_right_x, upper_right_y], [lower_right_x, lower_right_y]]], dtype=np.int32)
	# fill the ROI so it doesn't get wiped out when the mask is applied
	if image.shape == 3:
		channel_count = image.shape[2] #get the image channel
		ignore_mask_color = [255, 255, 255]
		mask = cv2.fillPoly(mask, roi_corners, ignore_mask_color)
		#print(mask)
		#apply the mask
		masked_image = cv2.bitwise_and(image, mask)
	else:
		ignore_mask_color = 255
		mask = cv2.fillPoly(mask, roi_corners, ignore_mask_color)
		#apply the mask
		#print(mask)
		masked_image = cv2.bitwise_and(image, mask)
	#Checkpoint
	#cv2.imshow('image', masked_image)
	return masked_image
if __name__ == '__main__':
	#read_video()
	
	pathIn = '/Users/dai/Downloads/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_00'
	pathOut = './'
	fps = 10
	cvt_frame_to_video(pathIn, pathOut, fps)
	


