import glob 
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from moviepy.editor import *
from IPython.display import HTML
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

def process_frame(image):
	#undistorting
	undistorted_image = undistort_image(image)

	#binary thresholding
	hls_thresh_image = hls_thresh(undistorted_image, channel=hls_thresh_channel, thresh=hls_thresh_threshold)
	abs_sobel_thresh_image = abs_sobel_thresh(undistorted_image, orient=abs_sobel_orient, sobel_kernel = abs_sobel_kernel, thresh= abs_sobel_threshold)
	combined_image = combine_threshs(hls_thresh_image, abs_sobel_thresh_image)
	windowed_image = filterf(combined_image)

	#perspective transform
	persp_transform_image = transform_image(windowed_image, M, img_size)

	#lanes
	out_img, ploty, left_fit, left_fitx, leftx_base, right_fit, right_fitx, rightx_base = find_lanes(persp_transform_image)

	#final image
	result = final_image(image, persp_transform_image, ploty, leftx_base, left_fit, left_fitx, rightx_base, right_fit, right_fitx, Minv)
	
	#plotting
	#plt.imshow(result)
	#plt.savefig('test_image_8_annotated')

	return result

#process_frame(test_images[7])


def process_video(input_path, output_path):
	input_file = VideoFileClip(input_path)
	output_clip = input_file.fl_image(process_frame)
	output_clip.write_videofile(output_path, audio=False)

process_video('test_videos/test_video_4.mp4', 'test_videos_results/test_video_4_annotated.mp4')






