import numpy as np
import cv2

class LaneLine():
    def hard_reset(self):
        self.ROC = 0.0 
        self.base_line = 0.0 
        self.no_fits = 5  
        self.detected_lanes = False
        self.x_present = [[]]
        self.y_present = [[]] 
        self.x_data = [[]] 
        self.y_data = [[]] 
        self.fit_present = np.array([0, 0, 0]) 
        self.fit_xy = np.array([0, 0, 0]) 

        
    def __init__(self):
        self.hard_reset()
        
def reset_lanes():
    leftside_lane.hard_reset()
    rightside_lane.hard_reset()

    

lane_width = 3.7 
lane_len = 3.0 
width_px = 675
len_px = 83 

ypp = lane_len/len_px 
xpp = lane_width/width_px 

leftside_lane = LaneLine()
rightside_lane = LaneLine()

def undistortFrames(frames, K, D):
    
    image_output = cv2.undistort(frames, K, D, None, K)
    return image_output

def applyCLAHE(frames):
    
    image_histogram = np.copy(frames)
    image_histogram = cv2.cvtColor(image_histogram, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
    image_histogram[:,:,0] = clahe.apply(image_histogram[:,:,0])
    image_histogram = cv2.cvtColor(image_histogram, cv2.COLOR_LAB2RGB)
        
    return image_histogram

def applyWarpPerspective(frames, M):
    
    warped_image = cv2.warpPerspective(frames, M, (frames.shape[1], frames.shape[0]), flags=cv2.INTER_LINEAR)
    return warped_image

def applyPerspectiveTransform(frames):
    width = frames.shape[1]
    height = frames.shape[0]
    
    source = np.array([[150, height],
                [480, 480],
                [720, 480],
                [width, height]], np.float32)
    

    destination = np.array([[300, height],
                [300, 0],
                [980, 0],
                [980, height]], np.float32)

    H = cv2.getPerspectiveTransform(source, destination)
    H_inv = cv2.getPerspectiveTransform(destination, source)
    return H, H_inv


def colorBinaryAnalyser(frames):

    input_CBT = np.copy(frames)
    
    def color_select(frames, val, threshold=(0, 255)):
        channel = frames[:,:,val]
        output = np.zeros_like(channel)
        if threshold[0] == threshold[1]:
            output[(channel >= threshold[0]) & (channel <= threshold[1])] = 1
        else:
            output[(channel > threshold[0]) & (channel <= threshold[1])] = 1
        return output

    input_LAB = cv2.cvtColor(input_CBT, cv2.COLOR_RGB2LAB)

    yellow_L = color_select(input_LAB, 0, threshold=(130, 255))
    yellow_A = color_select(input_LAB, 1, threshold=(100, 150))
    yellow_B = color_select(input_LAB, 2, threshold=(145, 210))
    
    yellow_R = color_select(input_CBT, 0, threshold=(255, 255))
    yellow_G = color_select(input_CBT, 1, threshold=(180, 255))
    yellow_b = color_select(input_CBT, 2, threshold=(0, 170))
    binary_yellow = np.zeros_like(yellow_L)
    binary_yellow[((yellow_R == 1) & (yellow_G == 1) & (yellow_b == 1))|((yellow_L == 1) & (yellow_A == 1) & (yellow_B == 1))] = 1

    
    white_L = color_select(input_LAB, 0, threshold=(230, 255))
    white_A = color_select(input_LAB, 1, threshold=(120, 140))
    white_B = color_select(input_LAB, 2, threshold=(120, 140))

    white_R = color_select(input_CBT, 0, threshold=(100, 255))
    white_G = color_select(input_CBT, 1, threshold=(100, 255))
    white_b = color_select(input_CBT, 2, threshold=(200, 255))
    white = np.zeros_like(white_L)
    
    white[((white_R == 1) & (white_G == 1) & (white_b == 1)) | ((white_L == 1) & (white_A == 1) & (white_B == 1))] = 1
    
    binary_color = np.zeros_like(binary_yellow)
    binary_color[(binary_yellow == 1) | (white == 1)] = 1
    
    color_flt = binary_color.astype(np.float)
    color = cv2.blur(color_flt, (3, 3))
    color_blur = np.zeros_like(binary_color)
    color_blur[ (color_flt > 0.0) ] = 1

    return color_blur



def applySobelDenoise(frames, kernel=5, threshold=0.7):
    
    binary_float = frames.astype(np.float)
    binary_float = cv2.blur(binary_float, (kernel, kernel))
    binary_denoise = np.zeros_like(frames)
    binary_denoise[ (binary_float > threshold) ] = 1

    return binary_denoise
    
def applySobelThreshold(frames, kernel=3, threshold=(0, 255)):
   
    img_gray = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
    abs_sobel = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    output = np.zeros_like(scaled_sobel)
    output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
    
    
    return output

def calcSobelMagnitude(frames, kernel=3, threshold=(0, 255)):
    
    img_gray = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    output = np.zeros_like(gradmag)
    output[(gradmag >= threshold[0]) & (gradmag <= threshold[1])] = 1
        
    return output

def calcSobelDirection(frames, kernel=3, threshold=(0, np.pi/2), debug=False):
    
    img_gray = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    output =  np.zeros_like(absgraddir, dtype=np.uint8)
    output[(absgraddir >= threshold[0]) & (absgraddir <= threshold[1])] = 1
    
    
    return output

def combineSobelMagDir(magnitude, direction):

    output = np.zeros_like(magnitude)
    output[(magnitude == 1) & (direction == 1)] = 1
        
    return output

def combineGradientThreshold(img_gradx, img_magdir):
    
    binary_gradient = np.zeros_like(img_gradx)
    binary_gradient[ (img_gradx == 1) | (img_magdir == 1) ] = 1
    
    return binary_gradient


def combineColorGT(binary_color, binary_gradient):
    
    if (leftside_lane.detected_lanes == True) and (rightside_lane.detected_lanes == True):
        binary_final = binary_color
    else:
        binary_final = cv2.bitwise_and(binary_color, binary_gradient)

    return binary_final


def lanePrediction(warped_image):
    
    x_left_lane = []
    y_left_lane = []
    x_right_lane = []
    y_right_lane = []
    win_no = 8
    min_px = 50 
    histogram_peak = 2.2 
    new_margin = 0.5 
    new_lane_m = 0.2
    
    hist_peak_margin = np.int(histogram_peak / xpp)
    window_margin = np.int(new_margin / xpp)
    lane_margin = np.int(new_lane_m / xpp)
    

    output_image = np.dstack((warped_image, warped_image, warped_image))*255

    histogram = np.sum(warped_image[warped_image.shape[0]//4*3:,:], axis=0)
    if np.sum(histogram) == 0:
       
        histogram = np.sum(warped_image[warped_image.shape[0]//2:,:], axis=0)
    
    first_peak_x = np.argmax(histogram)
    look_right_x = min(first_peak_x + hist_peak_margin, warped_image.shape[1]-1)
    look_left_x = max(first_peak_x - hist_peak_margin, 1)
    right_of_first_peak = max(histogram[look_right_x:])
    left_of_first_peak = max(histogram[:look_left_x])
    if right_of_first_peak > left_of_first_peak:
        
        win_center_leftx_base = first_peak_x
        win_center_rightx_base = np.argmax(histogram[look_right_x:]) + look_right_x
    else:
        
        win_center_rightx_base = first_peak_x
        win_center_leftx_base = np.argmax(histogram[:look_left_x])

    
    win_height = np.int(warped_image.shape[0]/win_no)
    nonzero = warped_image.nonzero()
    nzy = np.array(nonzero[0])
    nzx = np.array(nonzero[1])
    
    
    momentumX_left = 0
    momentumX_right = 0
    centerX_present = win_center_leftx_base
    centerX_present_right = win_center_rightx_base

    
    pst_lane_width = centerX_present_right - centerX_present

    
    for window_i in range(win_no):
        win_center_leftx_prev = centerX_present 
        win_center_rightx_prev = centerX_present_right

        
        wy_low = warped_image.shape[0] - (window_i+1)*win_height
        wy_high = warped_image.shape[0] - window_i*win_height
        wxl_low = centerX_present - window_margin
        wxl_high = centerX_present + window_margin
        wxr_low = centerX_present_right - window_margin
        wxr_high = centerX_present_right + window_margin

        
        cv2.rectangle(output_image, (wxl_low,wy_low), (wxl_high,wy_high), (255,0,0), 2) 
        cv2.rectangle(output_image, (wxr_low,wy_low), (wxr_high,wy_high), (255,0,0), 2) 

        
        required_left_idx = ((nzy >= wy_low) & (nzy < wy_high)
                        & (nzx >= wxl_low) & (nzx < wxl_high)).nonzero()[0]
        required_right_idx = ((nzy >= wy_low) & (nzy < wy_high)
                        & (nzx >= wxr_low) & (nzx < wxr_high)).nonzero()[0]
        total_left = len(required_left_idx)
        total_right = len(required_right_idx)

        
        if (total_left > min_px) and (total_right > min_px):
            
            centerX_present = np.int(np.mean(nzx[required_left_idx]))
            centerX_present_right = np.int(np.mean(nzx[required_right_idx]))
            pst_lane_width = centerX_present_right - centerX_present
        elif (total_left < min_px) and (total_right > min_px):
            
            centerX_present_right = np.int(np.mean(nzx[required_right_idx]))
            centerX_present = centerX_present_right - pst_lane_width
        elif (total_left > min_px) and (total_right < min_px):
            
            centerX_present = np.int(np.mean(nzx[required_left_idx]))
            centerX_present_right = centerX_present + pst_lane_width
        elif (total_left < min_px) and (total_right < min_px):
    
            centerX_present = win_center_leftx_prev + momentumX_left
            centerX_present_right = win_center_rightx_prev + momentumX_right

        momentumX_left = (centerX_present - win_center_leftx_prev)
        momentumX_right = (centerX_present_right - win_center_rightx_prev)

        wxl_low = centerX_present - lane_margin
        wxl_high = centerX_present + lane_margin
        wxr_low = centerX_present_right - lane_margin
        wxr_high = centerX_present_right + lane_margin
        cv2.rectangle(output_image, (wxl_low,wy_low), (wxl_high,wy_high), (255,0,255), 2) 
        cv2.rectangle(output_image, (wxr_low,wy_low), (wxr_high,wy_high), (255,0,255), 2) 
        required_left_idx = ((nzy >= wy_low) & (nzy < wy_high)
                        & (nzx >= wxl_low) & (nzx < wxl_high)).nonzero()[0]
        required_right_idx = ((nzy >= wy_low) & (nzy < wy_high)
                        & (nzx >= wxr_low) & (nzx < wxr_high)).nonzero()[0]

        x_left_lane.append(nzx[required_left_idx])
        y_left_lane.append(nzy[required_left_idx])
        x_right_lane.append(nzx[required_right_idx])
        y_right_lane.append(nzy[required_right_idx])

    leftside_lane.x_present = np.concatenate(x_left_lane)
    leftside_lane.y_present = np.concatenate(y_left_lane)
    rightside_lane.x_present = np.concatenate(x_right_lane)
    rightside_lane.y_present = np.concatenate(y_right_lane)
    
    if (len(leftside_lane.x_present) > 0) and (len(leftside_lane.x_present) == len(leftside_lane.y_present)):
        leftside_lane.fit_present = np.polyfit(leftside_lane.y_present, leftside_lane.x_present, 2)
    if (len(rightside_lane.x_present) > 0) and (len(rightside_lane.x_present) == len(rightside_lane.y_present)):
        rightside_lane.fit_present = np.polyfit(rightside_lane.y_present, rightside_lane.x_present, 2)
    
    
def existingLanePredict(warped_image):
    detected_margin = 0.3
    left_fit = leftside_lane.fit_present
    right_fit = rightside_lane.fit_present
    margin = np.int(detected_margin / xpp)

    nonzero = warped_image.nonzero()
    nzy = np.array(nonzero[0])
    nzx = np.array(nonzero[1])

    left_lane_inds = ((nzx > (left_fit[0]*(nzy**2) + left_fit[1]*nzy + left_fit[2] - margin)) 
                    & (nzx < (left_fit[0]*(nzy**2) + left_fit[1]*nzy + left_fit[2] + margin))) 
    right_lane_inds = ((nzx > (right_fit[0]*(nzy**2) + right_fit[1]*nzy + right_fit[2] - margin)) 
                     & (nzx < (right_fit[0]*(nzy**2) + right_fit[1]*nzy + right_fit[2] + margin)))  

    leftside_lane.x_present = nzx[left_lane_inds]
    leftside_lane.y_present = nzy[left_lane_inds] 
    rightside_lane.x_present = nzx[right_lane_inds]
    rightside_lane.y_present = nzy[right_lane_inds]

    if (len(leftside_lane.x_present) > 0) and (len(leftside_lane.x_present) == len(leftside_lane.y_present)):
        leftside_lane.fit_present = np.polyfit(leftside_lane.y_present, leftside_lane.x_present, 2)
    if (len(rightside_lane.x_present) > 0) and (len(rightside_lane.x_present) == len(rightside_lane.y_present)):
        rightside_lane.fit_present = np.polyfit(rightside_lane.y_present, rightside_lane.x_present, 2) 

def validLane(warped_image, debug=False):


    left_confirmed = True
    right_confirmed = True
    min_lane_width = 0
    max_lane_width = 0
    Dmin_px = 50 
    Dmax_px = 70000 
    Lwidth_min = 1.0 
    Lwidth_max = 6.0 
    left_fit = leftside_lane.fit_present
    right_fit = rightside_lane.fit_present
    left_x_num = len(leftside_lane.x_present)
    left_y_num = len(leftside_lane.y_present)
    right_x_num = len(rightside_lane.x_present)
    right_y_num = len(rightside_lane.y_present)
    lane_width_min_pix = np.int(Lwidth_min / xpp)
    lane_width_max_pix = np.int(Lwidth_max / xpp)

    evaluate_y = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0] )
    leftx_eval = left_fit[0]*evaluate_y**2 + left_fit[1]*evaluate_y + left_fit[2]
    rightx_eval = right_fit[0]*evaluate_y**2 + right_fit[1]*evaluate_y + right_fit[2]
    min_lane_width = min(rightx_eval - leftx_eval)
    max_lane_width = max(rightx_eval - leftx_eval)
    
    if min_lane_width < lane_width_min_pix:
        left_confirmed = False
        right_confirmed = False

    if max_lane_width > lane_width_max_pix:
        left_confirmed = False
        right_confirmed = False
     
    if ((left_x_num != left_y_num)
        or (left_x_num < Dmin_px) 
        or (left_x_num > Dmax_px)):
        left_confirmed = False
        right_confirmed = False
        
    if ((right_x_num != right_y_num)
        or (right_x_num < Dmin_px)
        or (right_x_num > Dmax_px)):
        left_confirmed = False
        right_confirmed = False
    
    leftside_lane.detected_lanes = left_confirmed
    rightside_lane.detected_lanes = right_confirmed

def updateLanes():
    if leftside_lane.detected_lanes == True and rightside_lane.detected_lanes == True:
        while len(leftside_lane.x_data) >= leftside_lane.no_fits:
            leftside_lane.x_data.pop(0)
            leftside_lane.y_data.pop(0)
        while len(rightside_lane.x_data) >= rightside_lane.no_fits:
            rightside_lane.x_data.pop(0)
            rightside_lane.y_data.pop(0)

        leftside_lane.x_data.append(leftside_lane.x_present)
        leftside_lane.y_data.append(leftside_lane.y_present)
        rightside_lane.x_data.append(rightside_lane.x_present)
        rightside_lane.y_data.append(rightside_lane.y_present)

        x_left_lane = np.concatenate(leftside_lane.x_data)
        y_left_lane = np.concatenate(leftside_lane.y_data)
        x_right_lane = np.concatenate(rightside_lane.x_data)
        y_right_lane = np.concatenate(rightside_lane.y_data)

        leftside_lane.fit_xy = np.polyfit(y_left_lane, x_left_lane, 2)
        rightside_lane.fit_xy = np.polyfit(y_right_lane, x_right_lane, 2) 

def laneColor(warped_image):
    left_fit = leftside_lane.fit_xy
    right_fit = rightside_lane.fit_xy
    left_x = leftside_lane.x_present
    left_y = leftside_lane.y_present
    right_x = rightside_lane.x_present
    right_y = rightside_lane.y_present

    plot_y = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0] )
    left_fitx = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
    right_fitx = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

    lane_dimensions = np.dstack((warped_image, warped_image, warped_image))*0

    points_left = np.array([np.transpose(np.vstack([left_fitx, plot_y]))])
    points_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, plot_y])))])
    points = np.hstack((points_left, points_right))

    if leftside_lane.detected_lanes == True and rightside_lane.detected_lanes == True:
        lane_color = (255,51,51)
    else:
        lane_color = (0, 50, 50)
    cv2.fillPoly(lane_dimensions, np.int_([points]), lane_color)
    lane_dimensions[left_y, left_x] = [0, 128, 0] #green
    lane_dimensions[right_y, right_x] = [255, 0, 0] #red
        
        
    return lane_dimensions

def calcROC(frames):
    leftx = np.concatenate(leftside_lane.x_data)
    lefty = np.concatenate(leftside_lane.y_data)
    rightx = np.concatenate(rightside_lane.x_data)
    righty = np.concatenate(rightside_lane.y_data)
    evaluate_y = frames.shape[0] # bottom y val
    
    if (len(leftx) > 0) and (len(rightx) > 0):
        curve_Lfit = np.polyfit(lefty*ypp, leftx*xpp, 2)
        curve_Rfit = np.polyfit(righty*ypp, rightx*xpp, 2)
        left_curve = (((1 + (2*curve_Lfit[0]*evaluate_y*ypp + curve_Lfit[1])**2)**1.5)
                          / np.absolute(2*curve_Lfit[0]))
        right_curve = (((1 + (2*curve_Rfit[0]*evaluate_y*ypp + curve_Rfit[1])**2)**1.5)
                          / np.absolute(2*curve_Rfit[0]))

        leftside_lane.ROC = left_curve
        rightside_lane.ROC = right_curve
        Lpoly = np.poly1d(curve_Lfit)
        Rpoly = np.poly1d(curve_Rfit)
        midx = np.int(frames.shape[1]/2)*xpp

        leftside_lane.base_line = midx - Lpoly(evaluate_y*ypp)
        rightside_lane.base_line = Rpoly(evaluate_y*ypp) - midx



def displayROC_offset(image_undist, lane_lanes, M_inv):
    def textOnImage(image, text, position):
        cv2.putText(image, text, position, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0,0,0), thickness=10)
        cv2.putText(image, text, position, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255,255,255), thickness=2)
        
    newwarp = cv2.warpPerspective(lane_lanes, M_inv, (lane_lanes.shape[1], lane_lanes.shape[0])) 
    textOnImage_result = cv2.addWeighted(image_undist, 1, newwarp, 1, 0)
    left_curve = leftside_lane.ROC
    right_curve = rightside_lane.ROC
    lane_offset = rightside_lane.base_line - leftside_lane.base_line
    if lane_offset < 0:
        side = 'right'
    else:
        side = 'left'
    textOnImage(textOnImage_result, 'Radius of curvature: L={:.0f} m, R={:.0f} m'
                                  .format(left_curve, right_curve), (50,100))
    textOnImage(textOnImage_result, 'Vehicle is {:.2f} m {} of center'
                                  .format(abs(lane_offset), side), (50,150))
    
    
    return textOnImage_result