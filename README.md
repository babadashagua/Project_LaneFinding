# **Report: Finding Lane Lines on the Road** 



---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Shortcomings and further improvements on the existing work


### Reflection

### 1. Lane Detection Pipeline

My lane detection pipeline is consisted of 5 steps. First, for each frame extracted from a video, I converted the image to grayscale, then I blur the image using Gaussian bluring to remove noise and use Canny edge detection algorithm to detect edges in the image. Since only the lane line are of interest, in the third step, I apply a trapezoidal mask on the image to allow process of lane regions only in the following steps. In the fourth step, I apply Hough transformation to the masked area to find the lane lines. Finally, the detected lane lines are drawn on the image. The following left image is from the img_test_results folder named "solidWhiteRight.jpg", the right image next to it shows the lane finding results after applying the developed pipeline.

<p float="left">
  <img src="/images/solidWhiteRight.jpg" width="400" title="Fig. 1 raw image solidWhiteRight.jpg"/>
  <img src="/images/solidWhiteRight_lanefinding.png" width="400" alt="Fig. 2 image solidWhiteRight.jpg with lane marking" /> 
</p>

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by seperate the result points after Hough transformation into two groups: left lane and right lane. For each lane group, I apply regression approach to find a single best-fitting line for the lane points. After applying the modified draw_lines() function, the pipeline is able to draw a single line on each lane line, as show in the following right image.

<p float="left">
  <img src="/images/solidWhiteRight.jpg" width="400" "/>
  <img src="/images/solidWhiteRight_solidline.png" width="400" /> 
</p>

I apply the developed pipeline to the video files located in the folder "test_videos", and the output videos are located in the folder "test_videos_output". The following left gif shows lane finding using single line on video "solidWhiteRight.mp4", the right gif shows lane finding using single line on video "solidYellowLeft.mp4". 

<p float="left">
  <img src="/images/solidWhiteRight_solidline.gif" width="400" "/>
  <img src="/images/solidYellowLeft_solidline.gif" width="400" /> 
</p>

### 2. Shortcomings with Current Pipeline

Shortcoming 1: to reduce detection error, I use a threshold to filter out the lines whose slope change is larger than the threshold. But it also results in slower convergence: there is a small mismatch between the drawn lane line and the real lane line.

### 3. Possible Improvements 

Improvement 1: deploy better filter to filter out the outlier detected lane lines, such as Kalman filter. 

