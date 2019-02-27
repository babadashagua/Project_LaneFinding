# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# to run for the first time. Guarantee python to use the correct opencv library
#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# functions used for lane finding, except the hough_lines() and lane_finding() functions, other helper functions I used are from 
# this project introduction.
def grayscale(img):
    
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    
def canny(img, low_threshold, high_threshold):
    
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# opencv use BGR format
def draw_lines(img, lines, color=[0, 0, 255], thickness=5):
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, slope):
    """
    `img` should be the output of a Canny transform.
    
    slope is an array that records past slope data to reduce lane detection errors
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    # seperate detected lane lines into two left lane and right lane 
    slope_array = np.empty(0)
    left_lane = np.zeros_like(lines)
    right_lane = np.zeros_like(lines)
    left_count = 0
    right_count = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (x2-x1) != 0:
                line_slope = (y2-y1)/(x2-x1)
                if line_slope > 0:
                    slope_array = np.append(slope_array, line_slope)
                    right_lane[right_count] = line
                    right_count += 1
                else:
                    slope_array = np.append(slope_array, line_slope)
                    left_lane[left_count] = line
                    left_count += 1
                
    # seperate left and right lane lines
    zero_i,zero_j,zero_k = np.where(left_lane == [[0,0,0,0]])
    zero_i = np.unique(zero_i)
    zero_i[::-1].sort()
    
    for i in range(len(zero_i)):
        left_lane = np.delete(left_lane, zero_i[i], 0)
        
    zero_i,zero_j,zero_k = np.where(right_lane == [[0,0,0,0]])
    zero_i = np.unique(zero_i)
    zero_i[::-1].sort()
    
    for i in range(len(zero_i)):
        right_lane = np.delete(right_lane, zero_i[i], 0)
    
    # regression for left and right lane:
    left_x = np.empty(0)
    left_y = np.empty(0)
    for line in left_lane:
        for x1,y1,x2,y2 in line:
            left_x = np.append(left_x, [x1, x2])
            left_y = np.append(left_y, [y1, y2])
            
    right_x = np.empty(0)
    right_y = np.empty(0)
    for line in right_lane:
        for x1,y1,x2,y2 in line:
            right_x = np.append(right_x, [x1, x2])
            right_y = np.append(right_y, [y1, y2])
            
    # find the y range for the lane line
    lane_upper = min(min(left_y), min(right_y))
    lane_lower = img.shape[1]
    
    # tolerance for detected lane line slope change
    slope_tolerance = 0.3
    z_left = np.polyfit(left_x, left_y, 1)
    f_left = np.poly1d(z_left)
    if len(slope) == 1:
        slope.append(f_left[1])
        slope.append(f_left[0])
    else:
        if abs(f_left[1] - slope[1]) < slope_tolerance:
            slope[1] = f_left[1] 
            slope[2] = f_left[0] 
        else:
            f_left[0] = slope[2]
            f_left[1] = slope[1]
            
    y1 = int(lane_upper)
    x1 = int((y1-f_left[0])/f_left[1])
    y2 = lane_lower
    x2 = int((y2-f_left[0])/f_left[1])
    cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),10)
    
    z_right = np.polyfit(right_x, right_y, 1)
    f_right = np.poly1d(z_right)
    if len(slope) == 3:
        slope.append(f_right[1])
        slope.append(f_right[0])
    else:
        if abs(f_right[1] - slope[1]) < slope_tolerance:
            slope[3] = f_right[1] 
            slope[4] = f_right[0] 
        else:
            f_right[0] = slope[4]
            f_right[1] = slope[3]
    y1 = int(lane_upper)
    x1 = int((y1-f_right[0])/f_right[1])
    y2 = lane_lower
    x2 = int((y2-f_right[0])/f_right[1])
    cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),10)
#    draw_lines(line_img, lines)
    return line_img, slope

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    
    return cv2.addWeighted(initial_img, α, img, β, γ)

# wrap the lane line detection and drawing on each frame as a function
def lane_finding(image, slope):

    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, 5) # later compare w/out gaussian blur 
    edges = canny(blur_gray, 50, 150)
    
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]),(460, 320), (490, 320), (imshape[1], imshape[0])]], dtype = np.int32)
    masked_img = region_of_interest(edges, vertices)
    line_image, slope = hough_lines(masked_img, 2, np.pi/180, 15, 35, 25, slope)
    
    line_edges = weighted_img(line_image, image)
    
    return line_edges, slope

cap = cv2.VideoCapture('test_videos/solidYellowLeft.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('test_videos_output/solidYellowLeft_solidline.mp4', 
                      fourcc, 25.0, (960, 540)) # 960*540
counter = 0
slope = []
while cap.isOpened():
    ret, frame = cap.read()
    counter += 1
    if not slope:
        slope.append(counter)

    if ret:
        frame_proc, slope = lane_finding(frame, slope)
        
        out.write(frame_proc)
        
    else:
        break
    
cap.release()
out.release()
print('Finished')

    
    
