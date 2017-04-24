## Advanced Lane Finding project
---
### Introduction 
I previously worked on a project called Lane Finding (on my Github) where I tried to detect lane lines on a dashcam video feed and annotated the video to  indicate the location of those lane markings. In this project, the end goal is the same but I'll be using more advanced techniques to make the model better at handling different lighting and weather situations and more complex curved lanes. I'm also making other improvements like lens distortion correction and frame-by-frame smoothing. 

The goals/steps I'll explain in depth are the following:
* Apply distortion correction to raw camera images to remove the effects that lenses have on images. 
* Use a combination of color spaces/transforms, edge detections algos/convolutional techniques, and filters, to create a thresholded binary image that shows the lane lines as clearly as possible. 
* Apply a perspective transform to get a rectified binary image (aka a "birds-eye view") to make it easier to determine lane curvature. 
* Detect lane pixels to find the lane boundaries.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output a visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

### Files and project navigation 
The project includes the following files:
* calibration.py, calibration_cal_corners, and calibration.p containing the calibration procedure, chessboard images with detected corners, and output coefficients.
* process_image.py containing the main pipelines
* image_gen.py containing helper functions
* video_annotated.mp4 contained the output video


### Camera Calibration
To do the camera calibration, I used a common technique where you compare images of chessboards, detect the corners, and compare the location of the corners to where they should be. 

More specifically, I started by preparing "object poinnts" which are the 3D (x,y,z) coordinates of the chessboard corners in the real world (z=0) and compare these with "image points" which I can detect using the `findChessboardCorners` function. Then, once I got the calibration and distortion coefficients, I used the the `cv2.undistort()` function to correct the test images and got the following results: 

### Creating a thresholded binary image
I did exploratory analysis to compare the effectiveness of various techniques. For each technique, I tried various kernels and thresholds. They included:
* absolute sobel threshold (in X and Y directions)
* magnitude sobel threshold
* directional threshold
* RGB thresholds
* HLS (hue/lightness/saturation) thresholds

Ultimately, I found that using a combination of the the HLS threshold and magnitude threshold works the best.

After applying these filters, I also utilized a filter/window to remove the area of the image where lane lines wouldn't be. 

### Perspective transform

The perspective transform changes the image such that you get a bird's eye view. This is important in order to determine lane curvature. 

The method for my perspective transform is called `transform(proc)`. The transformation is done by specifying "source points" and "destination points". Each set of points has 4 unique points and the transformation effectively specified how this 4-sided figure should look in the new space. The source points were manually chosen to be the trapezoid that makes up the main lane area. The destination points were also manually chosen to be a rectangle. My points were the following: 

```python
src = np.float32([(257, 685), (1050, 685), (583, 460),(702, 460)])
dst = np.float32([(200, 720), (1080, 720), (200, 0), (1080, 0)])
```
I then verified that the perspective transformation was working by drawing the source and destination points on a test image and its warped transformation, and ensuring the lines were parallel (left and right lane lines should always be parallel). 

### Identifying lane line pixels and fitting a polynomial

I started by creating a historgram for the buttom half of the transformed image and found the midpoint of the lane by taking the average of the two peaks. 

Then I utilized a sliding window approach to determine the location of the lanes as you go further away form the car. 

Once I had the windows and lane centers, I use the `np.polyfit` function to draw two second-order polynomials on the image to indicate the lane lines. 


### Radius of curvature and lane position relative to car 

The radius of curvature is the radius of a circle that touches a curve at a given point and has the same tangent and curvature at that point. I used standard formulas to calculate the radius on both the left and right lane lines. 

To calculate the lane position relative to the car I compared the center of the image (center of the car) to the midpoint between the left lane and right lane intersections with the bottom of the image. 

### Final image after undoing the transformation 

To undue the transform I used the `warpPerspective` function again but used the source and image points parameters in reverse order. After that I used the `fillPoly` function to color the are in between the lane lines in green. 

### Video pipeline
I created a separate file to process the video, called `process_video.py`. Here I used the moviepy library to read the video, edit it using the process function I defined, and save it. 
The output video file is called `video_annotated.mp4`.

### Discussion
The video pipeleine did a robust job of detecting lane lines, but it didn't perform too great on the challenge project.

In order to make the pipeline even more robust, I need to:
* Explore more ways to process images and apply better filters. There are so many combinations of color spaces and methods to find edges, that there are definitely ones out there which perform better.
* Test the pipelines on more videos to see if it performs well in fog, rain, snow etc.

