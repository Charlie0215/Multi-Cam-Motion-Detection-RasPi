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

def slid_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
	 xy_window=[64, 64], xy_overlap=(0.5, 0.5)):
	# If x and/or y start/stop positions not defined,
	# set to image size
	imgsizey = img.shape[0]
	imgsizex = img.shape[1]
	x_start_stop[0] = 0 if x_start_stop[0] is None else x_start_stop[0]
	x_start_stop[1] = imgsizex if x_start_stop[1] is None else x_start_stop[1]
	y_start_stop[0] = 0 if y_start_stop[0] is None else y_start_stop[0]
	y_start_stop[1] = imgsizey if y_start_stop[1] is None else y_start_stop[1]
	# Compute the span of the region to be searched
	sizex = x_start_stop[1] - x_start_stop[0]
	sizey = y_start_stop[1] - y_start_stop[0]
	# Compute the number of pixels per step in x/y
	stepx = int(xy_window[0] * xy_overlap[0])
	stepy = int(xy_window[1] * xy_overlap[0])
	# Compute the number of windows in x/y
	step_count_x = int(math.floor(1. * sizex / stepx)) - 1
	step_count_y = int(math.floor(1. * sizey / stepy)) - 1
	# Initializee a list to append window positions to
	window_list = []
	for i in range(step_count_y):
		for j in range(step_count_x):
			# Calculate each window position
			# Append window position to list
			window_list.append((
				(x_start_stop[0]+j*stepx, y_start_stop[0]+i*stepy),
				(x_start_stop[0]+j*stepx+xy_window[0], y_start_stop[0]+i*stepx+xy_window[1])
				))
	return window_list

def find_hot_boxes(image):
	window1 = {'x_limits': [None, None],
				'y_limits': [400, 640],
				'window_size': [128, 128],
				'overlap': [.5, .5]}

	window2 = {'x_limits': [32, None],
				'y_limits': [400, 640],
				'window_size': [96, 96],
				'overlap': [.5, .5]}

	window3 = {'x_limits': [412, 1280],
				'y_limits': [390, 540],
				'window_size': [80, 80],
				'overlap': [.5, .5]}
	windows_list = [slide_window(image, x_start_stop=window['x_limits'],
		y_start_stop=window['y_limits'], xy_window=wwindow['window_size'],
		xy_overlap=window['overlap']) for window in [window1, window2, window3]]

	output_image = np.copy(image)
	all_hot_windows = []

	#iterate over previously defined sliding windows
	
	for window in [window1, window2, window3]:
		windows = slide_window(
			output_image,
			x_start_stop=window['x_limits'],
			y_start_stop=window['y_limits'],
			xy_window=window['window_size'],
			xy_overlap=window['overlap']
		)

		hot_windows = []

		for window in windows:
		# Get test window from image
			test_image = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], 
				(64, 64), interpolation=cv2.INTER_AREA)
			features = extract_features(test_image, color_space=color_space, 
										spatial_size=spatial_size, hist_bins=hist_bins, 
										orient=orient, pix_per_cell=pix_per_cell, 
										cell_per_block=cell_per_block, 
										hog_channel=hog_channel, extract_spatial=extract_spatial, 
										extract_hist=extract_hist, extract_hog=extract_hog)
			'''
			detect feature
			'''
			if prediction == 1:
				hot_windows.append(window)
			all_hot_windows.extend(hot_windows)

		output_image = draw_boxes(output_image, hot_windows, color=(0, 0, 1), thick=4)
		
	return all_hot_windows, output_image

def extract_features(image, color_space='RGB', spatial_size=(32, 32),
						hist_bins=32, orient=9, 
						pix_per_cell=8, cell_per_block=2, hog_channel=0,
						extract_spatial=True, extract_hist=True, extract_hog=True):
	# Create a list to append feature vectors to
	features = []
	# apply color conversion if other than 'RGB'
	if color_space != 'RGB':
		feature_image = cv2.cvtColor (image, getattr(cv2, 'COLOR_RGB2' + color_space))
	else: feature_image = np.copy(image)      

	if extract_spatial:
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		features.append(spatial_features)
        
	if extract_hist:
		# Apply color_hist()
		hist_features = color_hist(feature_image, nbins=hist_bins)
		features.append(hist_features)
        
	if extract_hog:
        
		if color_space == 'GRAY':
			hog_features = get_hog_features(feature_image, orient, 
						pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            
		elif hog_channel == 'ALL':
			hog_features = [get_hog_features(feature_image[:,:,i], 
										orient, pix_per_cell, 
										cell_per_block, 
										vis=False, feature_vec=True)
							for i in range(3)]
			hog_features = np.ravel(hog_features)        
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
						pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		# Append the new feature vector to the features list
		features.append(hog_features)
        
	return np.concatenate(features)



if __name__ == '__main__':
	#read_video()
	
	pathIn = '/Users/dai/Downloads/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_00'
	pathOut = './'
	fps = 10
	cvt_frame_to_video(pathIn, pathOut, fps)
	


