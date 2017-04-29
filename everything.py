import glob 
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import cv2
from image_gen import *


#--------------------------------------------------------------------
#ORIGINALSs

originals = glob.glob('./test_images/test*.jpg')

#store images
images = []

#add images to vector
for original in originals:
	image = cv2.imread(original)
	images.append(image)


#--------------------------------------------------------------------
#UNDISTORT

#calibration values
calibration = pickle.load( open("calibration.p", "rb") )
mtx = calibration['mtx']
dist = calibration['dist']

#save undistorted
undistorted = []

for image in images:
	#undistort
	undist = cv2.undistort(image,mtx,dist,None,mtx)
	#append
	undistorted.append(undist)


#--------------------------------------------------------------------
#PROCESSING

#save processed
processed = []

for undist in undistorted:

	#process
	abs_sobel_thresh_1 = abs_sobel_thresh(undist, orient='x', sobel_kernel = 25, thresh= (30,100))
	mag_thresh_1 = mag_thresh(undist, sobel_kernel = 3,thresh= (60,100))
	dir_thresh_1 = dir_thresh(undist, sobel_kernel = 9, thresh= (-np.pi/4,np.pi/4))
	rgb_thresh_1 = rgb_thresh(undist, channel="r", thresh=(0,100))
	hls_thresh_1 = hls_thresh(undist, channel="s", thresh=(0,110))

	#combine thresholds
	combined = np.zeros_like(undist[:,:,0])
	combined[hls_thresh_1 == 1] = 1
	combined[abs_sobel_thresh_1 == 1] = 1

	#window
	windowed = filterf(combined)

	processed.append(windowed)




#--------------------------------------------------------------------
#TRANSFORMING

#save transformed
transformed = []

for proc in processed:
	img_size = (proc.shape[1], proc.shape[0])

	#trapazoid source points to rectangular destination points 
	src = np.float32([(257, 685), (1050, 685), (583, 460),(702, 460)])
	dst = np.float32([(200, 720), (1080, 720), (200, 0), (1080, 0)])

	#perspective tranform matrix
	M = cv2.getPerspectiveTransform(src, dst) 
	Minv = cv2.getPerspectiveTransform(dst, src) #inverse

	#transform
	trans = cv2.warpPerspective(proc, M, img_size)

	transformed.append(trans)


#--------------------------------------------------------------------
#LANES

for trans, image in zip(transformed, images):
	#create historgram for bottom half of trans
	hist = np.sum(trans[trans.shape[0]/2:,:], axis=0) 
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


	#step through windows one by one
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
	
	
	#curviature radius
	y_eval = np.max(trans[0])
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
	radi = [left_curverad, right_curverad]
	curvature_string = "Radius of Curvature: " + str(int(radi[0])) + ", " + str(int(radi[1]))
	
	#position from center
	pos = trans.shape[1]/2
	offset = abs(pos - (leftx_base + rightx_base)/2)
	location_string = "Vehicle Dist. from Center: " + str(offset)

	#unwarp
	warp_zero = np.zeros_like(trans).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	#Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
 	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (trans.shape[1], trans.shape[0])) 
    # Combine the result with the original image
	result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

	#text
	font = cv2.FONT_HERSHEY_SIMPLEX
	curvature_string = "Radius of Curvature: " + str(int(radi[0])) + ", " + str(int(radi[1]))
	location_string = "Vehicle Distance from Center: " + str(offset)

	#add text to image
	cv2.putText(result,curvature_string,(400,50), font, 1,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(result,location_string,(400,100), font, 1,(255,255,255),2,cv2.LINE_AA)

	f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,15))
	f.tight_layout

	ax1.imshow(out_img)
	ax1.set_title('Polynomial on lane lines', fontsize=12)
	ax1.set_xlim(0, 1280)
	ax1.set_ylim(720, 0)
	ax1.plot(left_fitx, ploty, color='yellow')
	ax1.plot(right_fitx, ploty, color='yellow')

	ax2.imshow(result)
	ax2.set_title('Full lane identification', fontsize=12)

	plt.show()






