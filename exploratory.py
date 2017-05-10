import glob 
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import cv2
from functions import * 

#IMAGES
images = read_images()

#PARAMS
img_size = (images[0].shape[1], images[0].shape[0])
abs_sobel_orient = 'x'
abs_sobel_kernel = 15
abs_sobel_threshold = (24,100)
mag_sobel_kernel = 15
mag_sobel_threshold = (20,100)
dir_sobel_kernel = 31
dir_sobel_threshold = (0, np.pi/2)
rgb_thresh_channel = 'r'
rgb_thresh_threshold = (130,255)
hls_thresh_channel = 's'
hls_thresh_threshold = (110,255)
hsv_thresh_channel = 'h'
hsv_thresh_threshold = (0,110)
YCrCb_thresh_channel = 'Y'
YCrCb_thresh_threshold = (110,250)
src = np.float32([(257, 685), (1050, 685), (583, 460),(702, 460)])
dst = np.float32([(200, 720), (1080, 720), (200, 0), (1080, 0)])
M = cv2.getPerspectiveTransform(src, dst) 
Minv = cv2.getPerspectiveTransform(dst, src)


#IMAGE TRANSFORMATIONS
undistorted_images = [undistort_image(image) for image in images]
abs_sobel_thresh_images = [abs_sobel_thresh(undistorted_image, orient=abs_sobel_orient, sobel_kernel = abs_sobel_kernel, thresh= abs_sobel_threshold) for undistorted_image in undistorted_images]
mag_thresh_images = [mag_thresh(undistorted_image, sobel_kernel = mag_sobel_kernel, thresh= mag_sobel_threshold) for undistorted_image in undistorted_images]
dir_thresh_images = [dir_thresh(undistorted_image, sobel_kernel = dir_sobel_kernel, thresh= dir_sobel_threshold) for undistorted_image in undistorted_images]
rgb_thresh_images = [rgb_thresh(undistorted_image, channel=rgb_thresh_channel, thresh=rgb_thresh_threshold) for undistorted_image in undistorted_images]
hls_thresh_images = [hls_thresh(undistorted_image, channel=hls_thresh_channel, thresh=hls_thresh_threshold) for undistorted_image in undistorted_images]
hsv_thresh_images = [hsv_thresh(undistorted_image, channel=hsv_thresh_channel, thresh=hsv_thresh_threshold) for undistorted_image in undistorted_images]
YCrCb_images = [YCrCb_thresh(undistorted_image, channel=YCrCb_thresh_channel, thresh=YCrCb_thresh_threshold) for undistorted_image in undistorted_images]
combined_images = [combine_threshs(hls_thresh, abs_sobel_thresh) for hls_thresh, abs_sobel_thresh in zip(hls_thresh_images, abs_sobel_thresh_images)]
windowed_images = [filterf(combined) for combined in combined_images]
birds_view_images = [transform_image(windowed_image, M, img_size) for windowed_image in windowed_images]
progress = [images, undistorted_images, abs_sobel_thresh_images, mag_thresh_images, dir_thresh_images, rgb_thresh_images, hls_thresh_images, hsv_thresh_images, YCrCb_images]


#PLOT CALIBRATION
def plot_calibration():
	no_corners = plt.imread('camera_calibration/camera_cal/calibration10.jpg')
	corners = plt.imread('camera_calibration/camera_cal_corners/calibration1_corners.jpg')
	labels = ['No corners', 'Corners']
	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize = (10,4))
	fig.tight_layout()
	ax1.imshow(no_corners)
	ax1.set_title(labels[0])
	ax1.axis('off')
	ax2.imshow(corners)
	ax2.set_title(labels[1])
	ax2.axis('off')
	plt.show()
	#plt.savefig('image.png', bbox_inches='tight', cmap='gray')
#plot_calibration()


#PLOT PROGRESS
def plot_progress(progress, test_image_number):
	labels = ['Original', 'Undistorted', 'Abs. Sobel Thresh.', 'Mag. Sobel Thresh.', 'Dir. Sobel Thresh.', 'RGB Thresh.', 'HLS Thresh.', 'HSV Thresh.', 'YCrCb Thresh']
	fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (15,10))
	axes = axes.ravel()
	fig.tight_layout()

	for ax, transformation, label in zip(axes, progress, labels):
		ax.imshow(transformation[test_image_number], cmap='gray')
		ax.set_title(label)
		ax.axis('off')
	#plt.show()
	plt.savefig('image.png', bbox_inches='tight', cmap='gray')
#plot_progress(progress, 0)


#PLOT ALL
def plot_all(images = [], birds_view_images = [], option = 'transformation'):
	labels = ["Yellow and white", "Straight white", "Poor lighting", "Yellow and white curved", "Yellow and white curved 2", "Very poor lighting 1", "Very poor lighting 2", "Very poor lighting 3"]
	fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,6))
	axes = axes.ravel()
	fig.tight_layout()

	if option == 'transformation':
		for ax, image, label in zip(axes, images[:4], labels[:4]):
			ax.imshow(image, cmap='gray')
			ax.set_title(label)
			ax.axis('off')

	elif option == 'lanes':
		for ax, birds_view_image, label in zip(axes, birds_view_images[:4], labels[:4]):
			out_img, ploty, left_fit, left_fitx, leftx_base, right_fit, right_fitx, rightx_base = find_lanes(birds_view_image)
			ax.imshow(out_img)
			ax.set_title(label)
			ax.plot(left_fitx, ploty, color='yellow')
			ax.plot(right_fitx, ploty, color='yellow')
			ax.axis('off')

	elif option == 'final':
		for ax, image, birds_view_image,label in zip(axes, images, birds_view_images[:4], labels[:4]):
			out_img, ploty, left_fit, left_fitx, leftx_base, right_fit, right_fitx, rightx_base = find_lanes(birds_view_image)
			result = final_image(image, birds_view_image, ploty, leftx_base, left_fit, left_fitx, rightx_base, right_fit, right_fitx, Minv)
			ax.imshow(result)
			ax.set_title(label)
			ax.axis('off')
	#plt.show()
	plt.savefig('readme_assets/image.png', bbox_inches='tight', cmap='gray')

#plot_all(images = birds_view_images, option = 'transformation') #plot transformations
#plot_all(images = images, birds_view_images = birds_view_images, option = 'lanes') #plot lane lines images
#plot_all(images = images, birds_view_images = birds_view_images, option = 'final') #plot final images


