## Advanced Lane Finding project
---
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

### Files and project navigation 
The project includes the following files:
* test_images and test_videos contain testing data.
* test_images_results and test_videos_results are folders that contain testing data with predicated lane lines.
* functions.py contains transformation functions and helper functions.
* exploratory.py contains parameters and methods to plot test images and transformations.
* pipeline.py contains the video processing pipeline and `process_frame(image)` function which is used on each frame. 
* The folder camera_calibration which includes calibration.py (script to do the calibration), calibration.p (calibration results/params), camera_cal (original images used for calibration), camera_cal_corners (original images with corners detected)


### Creating a thresholded binary image
A thresholded binary image an image that only has 2 types of pixels - pixels which make up the lane and pixels which don't. The idea is that you want to start by removing all noise, before trying to detect the lanes.


**Camera Calibration**

Most cameras distort images in some way. Although the effects are usually minor, it's important that we account for it so that we can later calculate lane curvature correctly. 

To determine the extent of lens distortion, I used a common technique to determine a transformation function that can be used to undistort an image. The technique works by taking images of chess boards, specifying how many chessboard corners are on the images, and using the OpenCV function `cv2.findChessboardCorners` to compare the position of where the corners should be vs. where they are found on the image. More specifically, I prepared "object points" which are the 3D (x,y,z) coordinates of the chessboard corners in the real world (z=0) and compared these with "image points" which I detected with the function above. Using the image points and object points, I could then use the OpenCV function `cv2.calibrateCamera()` to get a set of coefficients to undistort any other image. Finally I used `cv2.undistort()` on my test images. Note that the distortion mostly impacted the edges of the images. 


**Edge detection**
As you can see in the image above, I made use of 3 different edge detection techniques - the absolute Sobel threshold, the magnitude Sobel threshold, and the directional Sobel threshold. 







I did exploratory analysis to compare the effectiveness of various techniques. For each technique, I tried various kernels and thresholds. They included:
* absolute sobel threshold (in X and Y directions)
* magnitude sobel threshold
* directional threshold
* RGB thresholds
* HLS (hue/lightness/saturation) thresholds

Ultimately, I found that using a combination of the the HLS threshold and magnitude threshold works the best.

After applying these filters, I also utilized a filter/window to remove the area of the image where lane lines wouldn't be. 

![alt text][image1]


**Original images**
![alt text][image2]


**Multi-threshold binary image**
![alt text][image3]


**Region of interest**
![alt text][image4]


**Perspective transform**

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

