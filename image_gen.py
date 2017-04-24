import glob 
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import cv2

#---------------------------------------------------------
#thresholding functions

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
        threshold_channel = image[:,:,0] * 255
    if channel=="g":
        threshold_channel = image[:,:,1] * 255
    if channel=="b":
        threshold_channel = image[:,:,2] * 255

    rgb_threshold = np.zeros_like(threshold_channel)
    rgb_threshold[(threshold_channel > thresh[0]) & (threshold_channel < thresh[1])] = 1
    return rgb_threshold

def hls_thresh(image, channel="h", thresh=(0, 50)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    if channel=="h":
        threshold_channel = hls[:,:,0] * 255
    if channel=="l":
        threshold_channel = hls[:,:,1] * 255
    if channel=="s":
        threshold_channel = hls[:,:,2] * 255

    
    hls_threshold = np.zeros_like(threshold_channel)
    hls_threshold[(threshold_channel > thresh[0]) & (threshold_channel < thresh[1])] = 1
    return hls_threshold

def filterf(image):
	height, width = image.shape[0], image.shape[1]
	bl = (width / 2 - 450, height - 50)
	br = (width / 2 + 450, height - 50)
	tl = (width / 2 - 50, height / 2 + 85)
	tr = (width / 2 + 50, height / 2 + 85)

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
	
	img_window = image
	img_window[mask == False] = 0
	return img_window



#---------------------------------------------------------------
#exploratory research
def exploratory():
	#read image
	image = plt.imread('./test_images/undistorted2.jpg')

	#processing functions
	abs_sobel_thresh_1 = abs_sobel_thresh(image, orient='x', sobel_kernel = 25, thresh= (30,100))
	mag_thresh_1 = mag_thresh(image, sobel_kernel = 3,thresh= (60,100))
	dir_thresh_1 = dir_thresh(image, sobel_kernel = 9, thresh= (-np.pi/4,np.pi/4))
	rgb_thresh_1 = rgb_thresh(image, channel="r", thresh=(0,100))
	hls_thresh_1 = hls_thresh(image, channel="s", thresh=(0,110))
	

	#combined threshold function
	combined = np.zeros_like(image[:,:,0])
	combined[hls_thresh_1 == 1] = 1
	combined[mag_thresh_1 == 1] = 1

	#window
	windowed = window(combined)


	#plotting
	f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(1,8, figsize=(20,10))
	ax1.imshow(image)
	ax1.set_title('Original', fontsize=10)

	ax2.imshow(abs_sobel_thresh_1, cmap='gray')
	ax2.set_title('Abs. Sobel Thresh.', fontsize=10)

	ax3.imshow(mag_thresh_1, cmap='gray')
	ax3.set_title('Mag. Sobel Thresh.', fontsize=10)

	ax4.imshow(dir_thresh_1, cmap='gray')
	ax4.set_title('Dir. Sobel Thresh.', fontsize=10)

	ax5.imshow(rgb_thresh_1, cmap='gray')
	ax5.set_title('RGB Thresh.', fontsize=10)

	ax6.imshow(hls_thresh_1, cmap='gray')
	ax6.set_title('HLS Thresh.', fontsize=10)

	ax7.imshow(combined, cmap='gray')
	ax7.set_title('Combined Tresh.', fontsize=10)

	ax8.imshow(windowed, cmap='gray')
	ax8.set_title('Windowed', fontsize=10)


	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()

#exploratory()



#read original images and saved undistorted ones 
#--------------------------------------------------------

def undistort():
	originals = glob.glob('./test_images/test*.jpg')

	#calibration values
	calibration = pickle.load( open("calibration.p", "rb") )
	mtx = calibration['mtx']
	dist = calibration['dist']

	for idx, name in enumerate(originals):
		#read image
		image = cv2.imread(name)
		#undistort
		undistorted = cv2.undistort(image,mtx,dist,None,mtx)

		write_name = './test_images/undistorted'+str(idx)+'.jpg'
		cv2.imwrite(write_name, undistorted)

#undistort()


#preprocess images
#--------------------------------------------------------

def preprocess():

	undistorted = glob.glob('./test_images/undistorted*.jpg')

	for idx, name in enumerate(undistorted):
		#read image
		undist = plt.imread(name)

		#processing functions
		abs_sobel_thresh_1 = abs_sobel_thresh(undist, orient='x', sobel_kernel = 25, thresh= (30,100))
		mag_thresh_1 = mag_thresh(undist, sobel_kernel = 3,thresh= (60,100))
		dir_thresh_1 = dir_thresh(undist, sobel_kernel = 9, thresh= (-np.pi/4,np.pi/4))
		rgb_thresh_1 = rgb_thresh(undist, channel="r", thresh=(0,100))
		hls_thresh_1 = hls_thresh(undist, channel="s", thresh=(0,110))
		

		#combined threshold function
		combined = np.zeros_like(undist[:,:,0])
		combined[hls_thresh_1 == 1] = 1
		combined[mag_thresh_1 == 1] = 1

		#window
		windowed = window(combined)

		write_name = './test_images/preprocessed'+str(idx)+'.jpg'
		plt.imsave(write_name, windowed, cmap = 'gray')

#preprocess()





#transform images
#--------------------------------------------------------

def transform():
	preprocessed = glob.glob('./test_images/preprocessed*.jpg')

	for idx, name in enumerate(preprocessed):
		#read image
		img = plt.imread(name)
		img_size = (img.shape[1], img.shape[0])

		#trapazoid source points to rectangular destination points 
		src = np.float32([(257, 685), (1050, 685), (583, 460),(702, 460)])
		dst = np.float32([(200, 720), (1080, 720), (200, 0), (1080, 0)])

		#perspective tranform matrix
		M = cv2.getPerspectiveTransform(src, dst) 

		#transform
		transformed = cv2.warpPerspective(img, M, img_size)

		#write
		write_name = './test_images/transformed'+str(idx)+'.jpg'
		cv2.imwrite(write_name, transformed)

transform()



