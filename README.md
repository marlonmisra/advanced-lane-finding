## Advanced Lane Finding project


### Introduction 
I previously worked on a [lane finding project](https://github.com/marlonmisra/lane-finding) where I built a simple pipeline that detects lanes on front-facing car video footage. The goal remains the same, but here I'm using more advanced techniques to make the model more robust. Robust in this context means that the pipeline should perform well in poorer lighting conditions, worse weather conditions, and is agnostic to the color of the lane lines. Rather than just predict linear lane lines, this time I'll also predict the curvature of the lanes. 

The goals/steps I'll explain in depth are: 
* Applying distortion correction to raw camera images to remove the effects that lenses have on images. 
* Exploring different color spaces, including RGB, HLS, HSV, and YCrCb, and choosing the combination that works best together.
* Using convolutional techniques like the Soble operator for edge detection. 
* Applying a perspective transform to get a birds-eye view of the lanes such that it's easier to determine lane curvature.
* Detecting pixels that belong the lane using a convolutional technique. 
* Using the location of the lane pixels to fit a polynomial that matches the curvature of the lane. 
* Outputting the curvature of the lane and vehicle position with respect to center.
* Warping the images back into the original space and visually identifying the lane markings and the lane area in between. 

[//]: # (Image References)

[image1]: ./readme_assets/transformations.png "Transformations"
[image2]: ./readme_assets/original_images.png "Original images"
[image3]: ./readme_assets/combined_images.png "Combined images"
[image4]: ./readme_assets/windowed_images.png "Windowed images"
[image5]: ./readme_assets/birdsview_images.png "Birdsview images"
[image6]: ./readme_assets/lanes_images.png "Lanes images"
[image7]: ./readme_assets/final_images.png "Final images"
[image8]: ./readme_assets/sobel.gif "Sobel"
[image9]: ./readme_assets/distortion.png "distortion"

### Files and project navigation 
The project includes the following files:
* test_images and test_videos contain testing data.
* test_images_results and test_videos_results are folders that contain testing data with predicated lane lines.
* functions.py contains transformation functions and helper functions.
* exploratory.py contains parameters and methods to plot test images and transformations.
* pipeline.py contains the video processing pipeline and `process_frame(image)` function which is used on each frame. 
* The folder camera_calibration which includes calibration.py (script to do the calibration), calibration.p (calibration results/params), camera_cal (original images used for calibration), camera_cal_corners (original images with corners detected)

### Undistortion

Most cameras distort images in some way. Although the effects are usually minor, it's important that we account for it so that we can later calculate lane curvature correctly. 

To determine the extent of lens distortion, I used a common technique to determine a transformation function that can be used to undistort an image. The technique works by taking images of chess boards, specifying how many chessboard corners are on the images, and using the OpenCV function `cv2.findChessboardCorners` to compare the position of where the corners should be vs. where they are found on the image. More specifically, I prepared "object points" which are the 3D (x,y,z) coordinates of the chessboard corners in the real world (z=0) and compared these with "image points" which I detected with the function above. Using the image points and object points, I could then use the OpenCV function `cv2.calibrateCamera()` to get a set of coefficients to undistort any other image. Finally I used `cv2.undistort()` on my test images. Note that the distortion mostly impacted the edges of the images. 

![alt text][image9]


```python
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
			cv2.drawChessboardCorners(cal_img, (nx, ny), corners, ret)
			write_path = "./camera_cal_corners/calibration" + str(idx) + "_corners.jpg"
			cv2.imwrite(write_path, cal_img)
			print("Saved corners for image ", str(idx))


#load arbitrary image for size
img  = cv2.imread('./camera_cal/calibration11.jpg')
img_size = (img.shape[1], img.shape[0])

#camera calibration coefficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

#save camera calibration results for later
calibration = {}
calibration['mtx'] = mtx
calibration['dist'] = dist
pickle.dump(calibration, open("./calibration.p", "wb"))
print("Saved calibration file")
```

```python
def undistort_image(image):
	calibration = pickle.load( open("./camera_calibration/calibration.p", "rb") )
	mtx = calibration['mtx']
	dist = calibration['dist']
	undist = cv2.undistort(image,mtx,dist,None,mtx)
	return undist
```

### Binary thresholds 

**Approach**

A thresholded binary image an image that only has 2 types of pixels - pixels which make up the lane and pixels which don't. To create a thresholded binary image, I did 2 things - (1) detect lane pixels by using edge detection, and (2) detect lane pixels by using different color spaces. Below is an image that shows all the transformations I looked at for exploratory analysis. 

![alt text][image1]

**Edge detection**

I made use of 3 different edge detection techniques - the absolute Sobel threshold, the magnitude Sobel threshold, and the directional Sobel threshold. For the absolute Sobel threshold, you can either use a kernel to detect changes in the X direction or Y direction. I found the X direction to work best because lane lines are most visible as you look at the image from left to right. The way the absolute Sobel X operator works is that it defines a small NxN matrix which is moved across the whole image. The NxN matrix has values such that when you multiply them by the values of the image below you get a number which tells you about the gradient. For this to work the Soble operator is defined as follows for a small kernel of 2. 

![alt text][image8]

I also used a magnitude Sobel threshold which uses a combination of the absole sobel threshold in the X and Y direction. I found this one to work well, but not as well as the simple Sobel X operator. 

Lastly, I used the directional Sobel threshold, where I calculated the arctan of (Sobel X/Sobel Y) to constrain the search for gradients in a specific direction. I found this technique to be unnecessarily complex compared to the simple Sobel X operator. 

```python
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
```

**Color transforms**
I experimented with several color spaces and parameters to see if any of them were particularly useful. In total I looked at 4 space - the RGB (reg/green/blue) space, the HLS (hue/lightness/saturation) space, the HSV (hue/saturation/value) space, and the YCrCb (luma/blue-difference chroma component)/(red-difference chroma component) space. For each space, I defined the function in a way such that I could set a lower and upper bound on the value I want to filter by. Overall, the HLS space gave the best results, and it was specifically the S or saturation component that worked most robustly in different lighting conditions. I'm only giving a code example of the HLS conversion function below because they were all implemented in the same way (OpenCV has simple functions to convert from RGB space to others. 

```python
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
```


### Combined thresholds

After experimenting with combinations of different transforms, I found that the absolute Sobel threshold (in the X direction) together created the best binary thresholded image. The parameters that worked best are below. 

abs_sobel_kernel = 15
abs_sobel_threshold = (24,100)
hls_thresh_channel = 's'
hls_thresh_threshold = (110,255)

![alt text][image2]

![alt text][image3]

```python
def combine_threshs(hls_thresh_1, abs_sobel_thresh_1):
	combined = np.zeros_like(hls_thresh_1)
	combined[hls_thresh_1 == 1] = 1
	combined[abs_sobel_thresh_1 == 1] = 1
	return combined
```

### Region of interest

Up until this point, the goal of the binary thresholding was more focused on identifying pixels that belonged the lanes rather than minimizing false positives. One way to effectively reduce false positives now is to apply a region of interest window that sets all pixels in non-important areas to 0. These are pixels on the far left and right, and near the top where the sky is. The region of interst area looks like a trapezoid. 

![alt text][image4]

```python
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
```


### Perspective transform

The perspective transform changes the image such that you get a bird's eye view. This is important in order to determine lane curvature. 

The method for my perspective transform is called `transform(proc)`. The transformation is done by specifying "source points" and "destination points". Each set of points has 4 unique points and the transformation effectively specified how this 4-sided figure should look in the new space. The source points were manually chosen to be the trapezoid that makes up the main lane area. The destination points were also manually chosen to be a rectangle. My points were the following: 

```python
src = np.float32([(257, 685), (1050, 685), (583, 460),(702, 460)])
dst = np.float32([(200, 720), (1080, 720), (200, 0), (1080, 0)])
```
I then verified that the perspective transformation was working by drawing the source and destination points on a test image and its warped transformation, and ensuring the lines were parallel (left and right lane lines should always be parallel). 

![alt text][image5]


### Identifying lane line pixels and fitting a polynomial

I started by creating a historgram for the buttom half of the transformed image and found the midpoint of the lane by taking the average of the two peaks. 

Then I utilized a sliding window approach to determine the location of the lanes as you go further away form the car. 

Once I had the windows and lane centers, I use the `np.polyfit` function to draw two second-order polynomials on the image to indicate the lane lines. 

![alt text][image6]


### Radius of curvature and lane position relative to car 

The radius of curvature is the radius of a circle that touches a curve at a given point and has the same tangent and curvature at that point. I used standard formulas to calculate the radius on both the left and right lane lines. 

To calculate the lane position relative to the car I compared the center of the image (center of the car) to the midpoint between the left lane and right lane intersections with the bottom of the image. 



### Final image after undoing the transformation 

To undue the transform I used the `warpPerspective` function again but used the source and image points parameters in reverse order. After that I used the `fillPoly` function to color the are in between the lane lines in green. 


![alt text][image7]


### Video pipeline
I created a separate file to process the video, called `process_video.py`. Here I used the moviepy library to read the video, edit it using the process function I defined, and save it. 
The output video file is called `video_annotated.mp4`.


### Discussion
The video pipeleine did a robust job of detecting lane lines, but it didn't perform too great on the challenge project.

In order to make the pipeline even more robust, I need to:
* Explore more ways to process images and apply better filters. There are so many combinations of color spaces and methods to find edges, that there are definitely ones out there which perform better.
* Test the pipelines on more videos to see if it performs well in fog, rain, snow etc.

