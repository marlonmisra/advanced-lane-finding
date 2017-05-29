## Advanced Lane Finding project


### Introduction 
I previously worked on a [lane finding project](https://github.com/marlonmisra/lane-finding) where I built a simple pipeline that detects lanes on front-facing car video footage. The goal remains the same, but here I'm using more advanced techniques to make the model more robust. Robust means that the pipeline should perform well in poorer lighting conditions, worse weather conditions, and is agnostic to the color of the lane lines. Rather than just predict linear lane lines, this time I'll also predict the curvature of the lanes. 

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

[image1]: ./readme_assets/distortion.png "distortion"
[image2]: ./readme_assets/transformations.png "Transformations"
[image3]: ./readme_assets/sobel.gif "Sobel"
[image4]: ./readme_assets/original_images.png "Original images"
[image5]: ./readme_assets/combined_images_no_label.png "Combined images no label"
[image6]: ./readme_assets/combined_images.png "Combined images"
[image7]: ./readme_assets/windowed_images_no_label.png "Windowed images no label"
[image8]: ./readme_assets/windowed_images.png "Windowed images"
[image9]: ./readme_assets/birdsview_images_no_label.png "Birdsview images no label"
[image10]: ./readme_assets/birdsview_images.png "Birdsview images"
[image11]: ./readme_assets/detections_no_label.png "Detections no label"
[image12]: ./readme_assets/detections.png "Detections"
[image13]: ./readme_assets/final_images_no_label.png "Final images no label"
[image14]: ./readme_assets/video.gif "Video"


### Files and project navigation 
The project includes the following files:
* test_images and test_videos contain testing data.
* test_images_results and test_videos_results are folders that contain testing data with predicated lane lines.
* functions.py contains transformation functions and helper functions.
* exploratory.py contains parameters and methods to plot test images and transformations.
* pipeline.py contains the video processing pipeline and `process_frame(image)` function which is used on each frame. 
* The folder camera_calibration which includes calibration.py (script to do the calibration), calibration.p (calibration results/params), camera_cal (original images used for calibration), camera_cal_corners (original images with corners detected)

### Undistortion

Most cameras distort images in some way. Although the effects are usually minor, it's important that we account for it so that we can later calculate lane curvature correctly. The effect of distortion is most visibile in the area I highlighted with the red box, near the bottom of the image. 

To determine the transformation function that undistorts an image, I used a common technique that relies the analysis of standard images. In the camera_calibration folder, 20 images of chess boards were provided each of which was taken from a different perspective. Since chess boards have a fixed number of intersections and a fixed structure (90 degree line intersections at each corner), they are great for determining transformation functions to reverse lens distortion. I'm leaving out the details of the implementation but they are easy to follow in the calibration/calibration.py file. In the end, the procedure yielded a set of coefficient (`mtx` and `dist` below) that can be used with the `cv2.undistort()` function. 

![alt text][image1]


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

Binary threshold images only have 2 types of pixels - pixels which make up what you care about and pixels which don't. To create a thresholded binary image that highlights the lanes, I did 2 things. First, I detected lane pixels by using various edge detection techniques. Second, I detected lane pixels by looking at different color spaces. Below is an image that shows all the transformations I looked at for exploratory analysis. I'll later explain how I used a combination of these thresholded binary images to create a final thresholded binary image that does a good job of including lane pixels and excluding other pixels. 

![alt text][image2]

**Edge detection**

I made use of 3 different edge detection techniques - the absolute Sobel threshold, the magnitude Sobel threshold, and the directional Sobel threshold. For the absolute Sobel threshold, you can either use a kernel to detect changes in the X direction or Y direction. I found that the X direction works best because lane lines are most visible as you look at the image from left to right. The absolute Sobel operator works by moving a NxN filter across the image and computing the dot product between the filter and values of the image. Depending on the values of the filter, the dot product represents the gradient at that point in the image. For example, the Sobel X  and Sobel Y operators are defined as follows for a filter size of 3. 

![alt text][image3]

I also used a magnitude Sobel threshold which uses a combination of the absole sobel threshold in the X and Y direction. I found this one to work well, but not as well as the standard Sobel X operator. Lastly, I also used the directional Sobel threshold, where I calculated the arctan of (Sobel X/Sobel Y) to constrain the search for gradients in a specific direction. This technique showed some promise, but I was unable to find parameters (kernel and threshold) that worked really well for this problem. 

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


**Color transforms**

I experimented with several color spaces and parameters to see if any of them were particularly useful. In total I looked at 4 spaces - the RGB (reg/green/blue) space, the HLS (hue/lightness/saturation) space, the HSV (hue/saturation/value) space, and the YCrCb (luma/blue-difference chroma component)/(red-difference chroma component) space. For each space, I defined the function in a way such that I could set a lower and upper bound on the value I want to filter by. Overall, the HLS space gave the best results, and it was specifically the S or saturation component that worked most robustly in different lighting conditions. I'm only giving a code example of the HLS conversion function below because they were all implemented in the same way (OpenCV has simple functions to convert from RGB space to others. 

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

After experimenting with combinations of different transforms, I found that the absolute Sobel threshold (in the X direction) and the HLS (specifically the saturation threshold) together resulted in the best binary thresholded image. The parameters that worked best are below. 

![alt text][image4]

![alt text][image5]

```python
abs_sobel_kernel = 15
abs_sobel_threshold = (24,100)
hls_thresh_channel = 's'
hls_thresh_threshold = (110,255)
```

```python
def combine_threshs(hls_thresh_1, abs_sobel_thresh_1):
	combined = np.zeros_like(hls_thresh_1)
	combined[hls_thresh_1 == 1] = 1
	combined[abs_sobel_thresh_1 == 1] = 1
	return combined
```

### Region of interest

Up until this point, the goal of the binary thresholding was more focused on identifying pixels that belonged to lane markings rather than minimizing false positives. One way to effectively reduce false positives is to apply a region of interest mask that sets all pixels outside of it to 0. These are pixels on the far left and right and near the top where the sky is. The region of interest area looks like a trapezoid and is defined below. 

![alt text][image6]

![alt text][image7]

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

The perspective transform changes the image such that you get a bird's eye top-down view. This was a necesary step because it is much easier to determine lane curvate from this perspective. Before being able to apply a perspective transform, you need to define 4 source points and 4 destination points - knowledge of how each of these 4 points change lets you apply the OpenCV `cv2.warpPerspective` function. In order to get the source and destination I looked at a straight lane image, defined 4 points that make up the lane, and set the destination points in a way such that they make a rectangle. This has to be true because from a top-down view straight lanes should be parallel. The M parameter below defines the transformation to birds-view space and the Minv parameter defines the transformation back to the original space which we have to use later. 

```python
src = np.float32([(257, 685), (1050, 685), (583, 460),(702, 460)])
dst = np.float32([(200, 720), (1080, 720), (200, 0), (1080, 0)])
M = cv2.getPerspectiveTransform(src, dst) 
Minv = cv2.getPerspectiveTransform(dst, src)
```
![alt text][image8]

![alt text][image9]


### Identifying lane line pixels and fitting a polynomial

In order to identify lane pixels, I started by creating a histogram for the bottom half of the transformed image and found the midpoint of the lane by taking the average of the two peaks. Then I utilized a sliding window approach to determine the location of the lanes as you go further away form the car. Once I had the windows and lane centers, I drew two second-order polynomials on the image to indicate the lane lines. 

![alt text][image10]

![alt text][image11]

```python
def find_lanes(trans):
	#create histogram for bottom half of trans
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
	
	return out_img, ploty, left_fit, left_fitx, leftx_base, right_fit, right_fitx, rightx_base
```


### Radius of curvature and lane position relative to car 

The radius of curvature is the radius of a circle that touches a curve at a given point and has the same tangent and curvature at that point. I used standard formulas to calculate the radius on both the left and right lane lines. To calculate the lane position relative to the car I compared the center of the image (center of the car) to the midpoint between the left lane and right lane intersections with the bottom of the image. 

```python
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
```


### Final image after undoing the transformation 

To get the final image, I warped the birds-view image back into the original space. And then I drew the lanes and filled the area in between using the `fillPoly` function. 

![alt text][image12]

![alt text][image13]

```python
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
```


### Video pipeline
In `pipeline.py`, there are two functions defined. The first, `process_frame(image)` applies the steps described above in sequence to a frame. The second function, `process_video(input_path, output_path)`, applies the processing function to each frame, and saves a video of the output file. 

![alt text][image14]

### Discussion
The video pipeline did a great job of detecting lane lines. It also worked well on videos where lighting conditions varied, and on videos where multiple sharp turns happened in sequence. 

There are several improvements I still want to make to this pipeline: 
* Making the pipeline more robust to changes in camera position by doing a transformation in the beginning that converts to a standard perspective. 
* Doing multi-frame smoothing so that the pipeline benefits from the knowledge it gains from previous frames.
* Doing color thresholding on all 3 color channels rather than just one.





