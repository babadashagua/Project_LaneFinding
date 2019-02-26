# **Finding Lane Lines on the Road** 



---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Shortcomings and further improvements on the existing work


### Reflection

### 1. Lane Detection Pipeline

My lane detection pipeline is consisted of 5 steps. First, for each frame extracted from a video, I converted the image to grayscale, then I blur the image using Gaussian bluring to remove noise and use Canny edge detection algorithm to detect edges in the image. Since only the lane line are of interest, in the third step, I apply a trapezoidal mask on the image to allow process of lane regions only in the following steps. In the fourth step, I apply Hough transformation to the masked area to find the lane lines. Finally, the detected lane lines are drawn on the image. 

![Figure 1](images/solidWhiteRight.jpg)

![Figure 2]（images/solidWhiteRight_lanefinding.png）

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
