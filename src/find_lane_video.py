#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#%matplotlib inline

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
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


def get_x( vx, vy, x1, y1, y_ref ):
    """
    Helper function for draw_lines
    Calculates 'x' matching: 2 points on a line, its slope, and a given 'y' coordinate.
    """
    m = vy / vx
    b = y1 - ( m * x1 )
    x = ( y_ref - b ) / m
    return x


def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
#    for line in lines:
#        for x1,y1,x2,y2 in line:
#            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    max_slope = .85
    min_slope = .2
    r_pts = []
    l_pts = [] 
    top_y = 300
    bot_y = img.shape[0]
    smooth = 6    #lower number = more discared frames.

    for line in lines:
        for x1,y1,x2,y2 in line:
            # 1, find slope
            slope = float((y2-y1)/(x2-x1))
        
            # 2, use sloap to split lanes into left and right.
            # a negative slope will be right lane
            if max_slope >= slope >= min_slope:
                
                # append all points to points array 
                r_pts.append([x1,y1])
                r_pts.append([x2,y2])
                
                # declare numpy array
                # fit a line with those points
                # TODO explore other options besides DIST_12
                # TODO compare to polyfit implementation
                r_seg = np.array(r_pts)
                [r_vx, r_vy, r_cx, r_cy] = cv2.fitLine(r_seg, cv2.DIST_L12, 0, 0.01, 0.01)
                
                # define 2 x points for right lane line
                r_top_x = get_x( r_vx, r_vy, r_cx, r_cy, top_y )
                r_bot_x = get_x( r_vx, r_vy, r_cx, r_cy, bot_y )
                
            elif -max_slope <= slope <= -min_slope:
                
                # append all points to points array 
                l_pts.append([x1,y1])
                l_pts.append([x2,y2])
                
                # declare numpy array
                # fit a line with those points
                # TODO add something to test if segment points not blank
                l_seg = np.array(l_pts)
                [r_vx, r_vy, r_cx, r_cy] = cv2.fitLine(l_seg, cv2.DIST_L12, 0, 0.01, 0.01)
                
                # define 2 x points for left lane line
                l_top_x = get_x( r_vx, r_vy, r_cx, r_cy, top_y )
                l_bot_x = get_x( r_vx, r_vy, r_cx, r_cy, bot_y )
                
    cv2.line(img, (int(l_bot_x), bot_y), (int(l_top_x), top_y), color, thickness)
    cv2.line(img, (int(r_bot_x), bot_y), (int(r_top_x), top_y), color, thickness)

            

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
#    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
#    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
#    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    edges = canny(blur_gray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 300), (490, 300), (imshape[1],imshape[0])]], dtype=np.int32)
        
    # Next we'll create a masked edges image using cv2.fillPoly()
#    mask = np.zeros_like(edges)   
#    ignore_mask_color = 255   

#    cv2.fillPoly(mask, vertices, ignore_mask_color)
#    masked_edges = cv2.bitwise_and(edges, mask)
    
    masked_edges = region_of_interest(edges, vertices)
    
#    print('imshape[0] is', imshape[0])
#    print('imshape[1] is', imshape[1])

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
#    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
#    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)

    # Iterate over the output "lines" and draw lines on a blank image
#    for line in lines:
#        for x1,y1,x2,y2 in line:
#            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
#    draw_lines(line_image, lines)    

    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
#    lines_edges = cv2.addWeighted(color_edges, 1, line_image, 1, 0) 
#    lines_edges = weighted_img(line_image, color_edges, 1, 1, 0)
    lines_edges = weighted_img(line_image, image, 1, 1, 0)
        
        
        
    result = lines_edges

    return result



video_output = '../test_videos_output/output_solidYellowLeft.mp4'
#video_output = '../test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip2 = VideoFileClip('../test_videos/solidYellowLeft.mp4').subclip(0,5)
clip = VideoFileClip('../test_videos/solidYellowLeft.mp4')
#clip = VideoFileClip('../test_videos/challenge.mp4').subclip(0,5)
video_clip = clip.fl_image(process_image)


video_clip.write_videofile(video_output, audio=False)

'''
%time video_clip.write_videofile(video_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))
'''
