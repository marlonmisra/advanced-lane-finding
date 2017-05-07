import glob
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#corners
nx, ny = 9, 6
channels = 3

#calibration image list
image_names = glob.glob('./camera_cal/calibration*.jpg')

#imgpoints and objpoints
imgpoints = [] #2D in image plane
objpoints = [] #3D in real life, all same
objp = np.zeros((ny * nx, channels), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) #x, y coordinates


#loop through
for idx, image_name in enumerate(image_names):
		cal_img = mpimg.imread(image_name)
		cal_img_gray = cv2.cvtColor(cal_img,cv2.COLOR_BGR2GRAY)

		#get corners
		ret, corners = cv2.findChessboardCorners(cal_img_gray, (nx, ny), None)

		#add to object and image points
		if ret == True:
			print("Got corners for image ", str(idx))
			imgpoints.append(corners)
			objpoints.append(objp)

			#draw and display corners
			#cv2.drawChessboardCorners(cal_img, (nx, ny), corners, ret)
			#write_path = "./camera_cal_corners/calibration" + str(idx) + "_corners.jpg"
			#cv2.imwrite(write_path, cal_img)
			#print("Saved corners for image ", str(idx))


#load image for reference
img  = cv2.imread('./camera_cal/calibration11.jpg')
img_size = (img.shape[1], img.shape[0])

#camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

#save camera calibration results for later
calibration = {}
calibration['mtx'] = mtx
calibration['dist'] = dist
pickle.dump(calibration, open("./calibration.p", "wb"))
print("Saved calibration file")



