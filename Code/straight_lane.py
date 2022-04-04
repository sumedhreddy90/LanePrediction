import matplotlib.pylab as plt
import numpy as np
import math
import cv2

# Source for homography.
src = np.float32([[410,335], [535, 334], [780, 479], [150, 496]])
# Destination for homography
dst = np.float32([[50, 0], [250, 0], [250, 500], [0, 500]])

slope_left = 0
slope_right = 0
leftc = [0, 0, 0]
rightc = [0, 0, 0]

# Performing Canny Edge Detection and Hough Transformation to detect lane_lines
def edgeDetection(frames):

    input_shape = frames.shape
    vertices = np.array([[(0,input_shape[0]), (9*input_shape[1]/20, 11*input_shape[0]/18), (11*input_shape[1]/20, 11*input_shape[0]/18), (input_shape[1],input_shape[0])]], dtype=np.int32)
   
    # converting to grayscale
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    # applying filter 
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # detecting edges using canny
    edges = cv2.Canny(blur, 25, 100)
    # applying mask
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(edges, mask)
    cropped_img=ROI(frames)
    required_image=cv2.bitwise_and(cropped_img,cropped_img,mask=masked)
    required_image=cv2.cvtColor(required_image, cv2.COLOR_BGR2GRAY)
    # Applying Hough Transform to detect lane_lines
    lane_lines = cv2.HoughLinesP(required_image, 1, np.pi/180, 14, np.array([]), minLineLength=30, maxLineGap=60)
    hough_lines = np.zeros((*required_image.shape, 3), dtype=np.uint8)

    return hough_lines,masked, lane_lines,edges

def ROI(frames):
    input_shape = frames.shape[:2]
    vertices = np.array([[(0,input_shape[0]), (9*input_shape[1]/20, 11*input_shape[0]/18), (11*input_shape[1]/20, 11*input_shape[0]/18), (input_shape[1],input_shape[0])]], dtype=np.int32)
    mask = np.zeros_like(frames).astype(np.uint8)
    cv2.fillPoly(mask, [vertices], (255,255,255))
    cropped_image= cv2.bitwise_and(mask,frames)
    return cropped_image

# Estimating bold and dashed lanes through histogram peaks
def histogramPeakLanes(frames):
    histogram_image=np.sum(frames[frames.shape[0]//2:,:], axis=0)
    mid=int(histogram_image.shape[0]/2)
    rx_i=np.argmax(histogram_image[mid:])+mid
    lx_i=np.argmax(histogram_image[:mid])
    nz_pix=histogram_image.nonzero()
    centering=int(frames.shape[1]/2)
    left_side=frames[:,:centering]
    right_side=frames[:,centering:]
    l_count = cv2.findNonZero(left_side)
    r_count=cv2.findNonZero(right_side)
 
    return lx_i,rx_i,nz_pix,l_count,r_count

def draw_lines(img, lines,color_left,color_right,thickness=13):
    global slope_left
    global slope_right
    global leftc
    global rightc

    weights = 0.9

    right_ys = []
    right_xs = []
    right_slopes = []

    left_ys = []
    left_xs = []
    left_slopes = []

    midpoint = img.shape[1] / 2

    bottom_of_image = img.shape[0]
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope, yint = np.polyfit((x1, x2), (y1, y2), 1)
            # Filter lines using slope and x position
            if .35 < np.absolute(slope) <= .85:
                if slope > 0 and x1 > midpoint and x2 > midpoint:
                    right_ys.append(y1)
                    right_ys.append(y2)
                    right_xs.append(x1)
                    right_xs.append(x2)
                    right_slopes.append(slope)
                elif slope < 0 and x1 < midpoint and x2 < midpoint:
                    left_ys.append(y1)
                    left_ys.append(y2)
                    left_xs.append(x1)
                    left_xs.append(x2)
                    left_slopes.append(slope)
   
    
    # Drawing right lane
    
    if right_ys:
        right_index = right_ys.index(min(right_ys))
        right_x1 = right_xs[right_index]
        right_y1 = right_ys[right_index]
        right_slope = np.median(right_slopes)
        if slope_right != 0:
            right_slope = right_slope + (slope_right - right_slope) * weights

        right_x2 = int(right_x1 + (bottom_of_image - right_y1) / right_slope)

        if slope_right != 0:
            right_x1 = int(right_x1 + (rightc[0] - right_x1) * weights)
            right_y1 = int(right_y1 + (rightc[1] - right_y1) * weights)
            right_x2 = int(right_x2 + (rightc[2] - right_x2) * weights)

        slope_right = right_slope
        rightc = [right_x1, right_y1, right_x2]
        cv2.line(img, (right_x1, right_y1), (right_x2, bottom_of_image), color_right, thickness)
        
        
    

    # Drawing left lane
    if left_ys:
        left_index = left_ys.index(min(left_ys))
        left_x1 = left_xs[left_index]
        left_y1 = left_ys[left_index]
        left_slope = np.median(left_slopes)
        if slope_left != 0:
            left_slope = left_slope + (slope_left - left_slope) * weights

        left_x2 = int(left_x1 + (bottom_of_image - left_y1) / left_slope)

        if slope_left != 0:
            left_x1 = int(left_x1 + (leftc[0] - left_x1) * weights)
            left_y1 = int(left_y1 + (leftc[1] - left_y1) * weights)

        slope_left = left_slope
        leftc = [left_x1, left_y1, left_x2]
        cv2.line(img, (left_x1, left_y1), (left_x2, bottom_of_image), color_left, thickness)


def resultsPlotter(frames,hough_lines,lane_lines,l_count,r_count):
   
    if  r_count.shape[0] > l_count.shape[0]:
        color_left = (0,0,255)
        left_text="Dashed Lines"
        color_right= (0,255,0)
        right_text="Solid Lines"
    else:
         print("left")
         color_left = (0,255,0)   
         left_text = "Solid Lines"
         color_right = (0,0,255)    
         right_text="Dashed Lines" 

    draw_lines(hough_lines,lane_lines,color_left,color_right)
    
    processed = cv2.addWeighted(frames, 0.8, hough_lines, 1, 0)
    
    cv2.putText(processed,left_text,(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)
    cv2.putText(processed,right_text,(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)

    
    return processed
if __name__ == "__main__":
    path = 'whiteline.mp4'
    input_video = cv2.VideoCapture(path)
    output_video='problem_2.mp4'
    output = cv2.VideoWriter(output_video,cv2.VideoWriter_fourcc(*'mp4v'), 15, (960,540))

    count=0
    while(input_video.isOpened()):
        success, images = input_video.read()
        if success:
            count+=1
            hough_lines,masked_edges, detected_lines,edges=edgeDetection(images)
            cv2.imshow("Canny Edges",edges)
            cv2.imwrite("2_Canny_Edges.jpg" , edges)
            homography, mask = cv2.findHomography( src,dst,cv2.RANSAC,5.0)
            warped_image = cv2.warpPerspective(masked_edges,homography,(300,600),flags=cv2.INTER_LINEAR)
            cv2.imwrite("2_warped.jpg" , warped_image)
            l,r,nxy,lcount,rcount=histogramPeakLanes(warped_image)
            final_output=resultsPlotter(images,hough_lines,detected_lines,lcount,rcount)
            if(count == 5):
                cv2.imwrite("2_straighlane_detected.jpg" , final_output)
            cv2.imshow("Output",final_output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
input_video.release()  
output_video.release()      
cv2.destroyAllWindows()