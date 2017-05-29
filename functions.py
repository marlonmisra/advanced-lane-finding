import glob 
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import cv2

def read_images():
	test_image_names = glob.glob('./test_images/test_image_*.jpg')
	test_images = []
	for test_image_name in test_image_names:
		image = plt.imread(test_image_name)
		test_images.append(image)
	return test_images

def undistort_image(image):
	calibration = pickle.load( open("./camera_calibration/calibration.p", "rb") )
	mtx = calibration['mtx']
	dist = calibration['dist']
	undist = cv2.undistort(image,mtx,dist,None,mtx)
	return undist

def undistort_image_rectangle(image):
	img_size = (image.shape[1], image.shape[0])
	undistorted_drawn = cv2.rectangle(np.copy(image), pt1=(0,img_size[1]), pt2=(img_size[0],img_size[1]-100), color=(255,0,0), thickness=3)
	return undistorted_drawn

def abs_sobel_thresh(image, orient = 'x', sobel_kernel=3, thresh = (0.7,5)):
	gray = cv2. cvtColor(image, cv2.COLOR_RGB2GRAY)

	if orient == 'x':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0 ,ksize = sobel_kernel)
	if orient =='y':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1 ,ksize = sobel_kernel)
	
	abs_sobel = np.absolute(sobel)
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

	abs_sobel_thresh = np.zeros_like(scaled_sobel)
	abs_sobel_thresh[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1

	return abs_sobel_thresh

def mag_thresh(image, sobel_kernel=3, thresh=(1, 5)):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0 ,ksize = sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1 ,ksize = sobel_kernel)

	sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
	scaled_magnitude = np.uint8(255*sobel_magnitude/np.max(sobel_magnitude))

	mag_thresh = np.zeros_like(scaled_magnitude)
	mag_thresh[(scaled_magnitude > thresh[0]) & (scaled_magnitude < thresh[1])] = 1

	return mag_thresh

def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2. cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0 ,ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1 ,ksize = sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    directions = np.arctan2(sobely, sobelx)

    dir_threshold = np.zeros_like(directions)
    dir_threshold[(directions > thresh[0]) & (directions < thresh[1])] = 1

    return dir_threshold

def rgb_thresh(image, channel="r", thresh=(0, 1)):
    if channel=="r":
        threshold_channel = image[:,:,0]
    if channel=="g":
        threshold_channel = image[:,:,1]
    if channel=="b":
        threshold_channel = image[:,:,2]

    rgb_threshold = np.zeros_like(threshold_channel)
    rgb_threshold[(threshold_channel > thresh[0]) & (threshold_channel < thresh[1])] = 1
    return rgb_threshold

def hls_thresh(image, channel="h", thresh=(0, 50)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    if channel=="h":
        threshold_channel = hls[:,:,0]
    if channel=="l":
        threshold_channel = hls[:,:,1]
    if channel=="s":
        threshold_channel = hls[:,:,2]

    hls_threshold = np.zeros_like(threshold_channel)
    hls_threshold[(threshold_channel > thresh[0]) & (threshold_channel < thresh[1])] = 1
    return hls_threshold

def hsv_thresh(image, channel="h", thresh=(0, 50)):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if channel=="h":
        threshold_channel = hsv[:,:,0]
    if channel=="s":
        threshold_channel = hsv[:,:,1]
    if channel=="v":
        threshold_channel = hsv[:,:,2]

    hsv_threshold = np.zeros_like(threshold_channel)
    hsv_threshold[(threshold_channel > thresh[0]) & (threshold_channel < thresh[1])] = 1
    return hsv_threshold

def YCrCb_thresh(image, channel="Y", thresh=(0, 50)):
    YCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    if channel=="Y":
        threshold_channel = YCrCb[:,:,0]
    if channel=="Cr":
        threshold_channel = YCrCb[:,:,1]
    if channel=="Cb":
        threshold_channel = YCrCb[:,:,2]

    YCrCb_threshold = np.zeros_like(threshold_channel)
    YCrCb_threshold[(threshold_channel > thresh[0]) & (threshold_channel < thresh[1])] = 1
    return YCrCb_threshold

def combine_threshs(hls_thresh_1, abs_sobel_thresh_1):
	combined = np.zeros_like(hls_thresh_1)
	combined[hls_thresh_1 == 1] = 1
	combined[abs_sobel_thresh_1 == 1] = 1
	return combined

def filterf(image):
	height, width = image.shape[0], image.shape[1]
	bl = (width / 2 - 480, height - 30)
	br = (width / 2 + 480, height - 30)
	tl = (width / 2 - 60, height / 2 + 76)
	tr = (width / 2 + 60, height / 2 + 76)

	fit_left = np.polyfit((bl[0], tl[0]), (bl[1], tl[1]), 1)
	fit_right = np.polyfit((br[0], tr[0]), (br[1], tr[1]), 1)
	fit_bottom = np.polyfit((bl[0], br[0]), (bl[1], br[1]), 1)
	fit_top = np.polyfit((tl[0], tr[0]), (tl[1], tr[1]), 1)

	# Find the region inside the lines
	xs, ys = np.meshgrid(np.arange(0, image.shape[1]), np.arange(0, image.shape[0]))
	mask = (ys > (xs * fit_left[0] + fit_left[1])) & \
           (ys > (xs * fit_right[0] + fit_right[1])) & \
           (ys > (xs * fit_top[0] + fit_top[1])) & \
           (ys < (xs * fit_bottom[0] + fit_bottom[1]))
	
	img_window = np.copy(image)
	img_window[mask == False] = 0

	return img_window

def transform_image(windowed_image, M, img_size):
	return cv2.warpPerspective(windowed_image, M, img_size)

def get_hist(img):
	hist = np.sum(img[int(img.shape[0]//2):,:], axis=0)
	#hist = np.sum(img[img.shape[0]/2:,:], axis=0)
	return hist

def find_lanes(trans):
	#create historgram for bottom half of trans
	hist = np.sum(trans[int(trans.shape[0]/2):,:], axis=0) 
	#output image to draw on + visualize
	out_img = np.dstack((trans, trans, trans))*255
	#peaks of left + right halves if hist	
	midpoint = np.int(hist.shape[0]/2)
	leftx_base = np.argmax(hist[:midpoint])
	rightx_base = np.argmax(hist[midpoint:]) + midpoint
	#number of sliding windows
	nwindows = 9
	#window height 
	window_height = np.int(trans.shape[0]/nwindows)

	#x and y positions of all nonzero pizels in img
	nonzero = trans.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	#current positions to be updated
	leftx_current = leftx_base
	rightx_current = rightx_base

	#width of windows +/- margin
	margin = 100
	#min number of pixels to recenter window
	minpix = 50

	#empty lists to receive left and right lane pixel indices
	left_lane_indices = []
	right_lane_indices = []

	#go through windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = trans.shape[0] - (window+1)*window_height
	    win_y_high = trans.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
	    # Append these indices to the lists
	    left_lane_indices.append(good_left_indices)
	    right_lane_indices.append(good_right_indices)
	    # If you found > minpix pixels, recenter next window on their mean position
	    if len(good_left_indices) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_indices]))
	    if len(good_right_indices) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_indices]))

	# Concatenate the arrays of indices
	left_lane_indices = np.concatenate(left_lane_indices)
	right_lane_indices = np.concatenate(right_lane_indices)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_indices]
	lefty = nonzeroy[left_lane_indices] 
	rightx = nonzerox[right_lane_indices]
	righty = nonzeroy[right_lane_indices] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	# Generate x and y values for plotting
	ploty = np.linspace(0, trans.shape[0]-1, trans.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_indices], nonzerox[left_lane_indices]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_indices], nonzerox[right_lane_indices]] = [0, 0, 255]
	
	return out_img, ploty, left_fit, left_fitx, leftx_base, right_fit, right_fitx, rightx_base

def curvature_radius(trans, left_fit, right_fit):
	y_eval = np.max(trans[0])
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
	radi = [left_curverad, right_curverad]
	curvature_string = "Radius of Curvature: " + str(int(radi[0])) + ", " + str(int(radi[1]))
	return curvature_string

def pos_from_center(trans, leftx_base, rightx_base):
	pos = trans.shape[1]/2
	offset = abs(pos - (leftx_base + rightx_base)/2)
	location_string = "Vehicle Dist. from Center: " + str(offset)
	return location_string

def final_image(image, persp_transform_image, ploty, leftx_base, left_fit, left_fitx, rightx_base, right_fit, right_fitx, Minv):
	#perspective transform back
	warp_zero = np.zeros_like(persp_transform_image).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	#Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
 	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (persp_transform_image.shape[1], persp_transform_image.shape[0])) 

    #Combine the result with the original image
	result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

	#info strings
	curvature_string = curvature_radius(persp_transform_image, left_fit, right_fit)
	location_string = pos_from_center(persp_transform_image, leftx_base, rightx_base)

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(result,curvature_string,(400,50), font, 1,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(result,location_string,(400,100), font, 1,(255,255,255),2,cv2.LINE_AA)
	return result













