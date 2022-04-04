import cv2
import numpy as np
import lane_detection

def laneDetetctionAlgorithm(frame):
    K = [[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
         [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    K = np.array(K)
    D = [[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]]
    D = np.array(D)

    #Method to undistort the frames by calibrating the camera
    undistorted_frames = lane_detection.undistortFrames(frame, K, D)
    sample_dist = undistorted_frames.copy()
    cv2.imwrite("undistorted.jpg", sample_dist)
    #Method to create CLAHE and apply to each frame
    CLAHE_frames = lane_detection.applyCLAHE(undistorted_frames)
    #Method to Calculate a perspective transform from four pairs of the corresponding points
    # Homography and Inverse Homographhy
    H, H_inv = lane_detection.applyPerspectiveTransform(undistorted_frames)
    
    #Method to Apply a perspective transformation to an image.
    #The function warpPerspective transforms the source image using the specified matrix M
    warped_frames = lane_detection.applyWarpPerspective(undistorted_frames, H)
    print_warp = warped_frames.copy()
    gray = cv2.cvtColor(print_warp, cv2.COLOR_BGR2GRAY)
    ret, thresh2 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    cv2.imwrite("warp.jpg", thresh2)
    # Method to extract binary yellow and white
    extract_colors = lane_detection.colorBinaryAnalyser(CLAHE_frames)
    cv2.imwrite("yellow_whote.jpg", extract_colors)
    binary_color_warped = lane_detection.applyWarpPerspective(extract_colors, H)
    
    if (lane_detection.leftside_lane.detected_lanes == False) or (lane_detection.rightside_lane.detected_lanes == False):
        gradient_warped = lane_detection.applySobelThreshold(warped_frames, threshold=(10, 255))
        gradient_warped = lane_detection.applySobelDenoise(gradient_warped, kernel=5, threshold=0.7)

        binary_mag = lane_detection.calcSobelMagnitude(undistorted_frames, kernel=3, threshold=(5, 255))
        binary_dir = lane_detection.calcSobelDirection(undistorted_frames, kernel=3, threshold=(0.5, 1.3))
        binary_magdir = lane_detection.combineSobelMagDir(binary_mag, binary_dir)
        binary_magdir = lane_detection.applySobelDenoise(binary_magdir, kernel=5, threshold=0.7)
        binary_magdir_warped = lane_detection.applyWarpPerspective(binary_magdir, H)
        binary_gradient_warped = lane_detection.combineGradientThreshold(gradient_warped, binary_magdir_warped)
        
    else:
        binary_gradient_warped = None

    final_lanes = lane_detection.combineColorGT(binary_color_warped, binary_gradient_warped)
    
    if (lane_detection.leftside_lane.detected_lanes == False) or (lane_detection.rightside_lane.detected_lanes == False):
        lane_detection.lanePrediction(final_lanes)
    else:
        lane_detection.existingLanePredict(final_lanes)
        
    lane_detection.validLane(final_lanes)
    lane_detection.updateLanes()
    lane_lines = lane_detection.laneColor(final_lanes)
    cv2.imwrite("lanes.jpg", lane_lines)
    lane_detection.calcROC(lane_lines)
    result = lane_detection.displayROC_offset(undistorted_frames, lane_lines, H_inv)
    cv2.imwrite("lanes_detected.jpg", result)
    final_image = show_output_video(frame, result, sample_dist, warped_frames,lane_lines)
    return final_image

def show_output_video(frames, result,undistorted_frames,warped_frames,lane_lines):
    
    height, width = 1080, 1920
    combined_image=np.zeros((height,width,3), np.uint8)
    cv2.putText(frames,'[1] Input Frames',(30,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 3, 0)
    cv2.putText(undistorted_frames,'[2] undistorted Frames',(30,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 3, 0)
    combined_image[360:720,1280:1920] = cv2.resize(undistorted_frames, (640,360), interpolation=cv2.INTER_AREA)
    combined_image[0:720,0:1280] = cv2.resize(result, (1280,720), interpolation=cv2.INTER_AREA)
    combined_image[0:360,1280:1920] = cv2.resize(frames, (640,360), interpolation=cv2.INTER_AREA)
    combined_image[720:1080,1280:1920] = cv2.resize(warped_frames, (640,360), interpolation=cv2.INTER_AREA)
    # black_footer= np.zeros((100, combined_image.shape[1], 3), np.uint8)
    # black_footer[:] = (112,128,144) 
    # combined_image=cv2.vconcat((combined_image,black_footer))
    combined_image[720:1080,640:1280] = cv2.resize(lane_lines, (640,360), interpolation=cv2.INTER_AREA)
    cv2.putText(combined_image,'[3] warped',(1333,788), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 3, 0)
    cv2.putText(combined_image,'',(61,802), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 3, 0)
    cv2.putText(combined_image,'[4] Detected Lanes',(608,787), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 3, 0)
    cv2.putText(combined_image,'[5] Lane Detection Algorithm',(40,675), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 3, 0)
    cv2.putText(combined_image,'[5] Lane Detection Algorithm',(532,1158), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, 0)
    cv2.putText(combined_image,'[1] Input Frames',(31,1116), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, 0)
    cv2.putText(combined_image,'[2] Detected Lanes',(524,1110), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, 0)
    cv2.putText(combined_image,'[3] warped',(989,1110), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, 0)
    cv2.putText(combined_image,'',(31,1163), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, 0)
    cv2.putText(combined_image,'[4] Detected Lanes',(1406,1109), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, 0)
    cv2.imwrite("combined_result.jpg", combined_image)
    return combined_image


if __name__ == '__main__':
  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  cap = cv2.VideoCapture('challenge.mp4')
  # Video writer for displaying output video
  output = cv2.VideoWriter('lane_curve_prediction.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (1920,1180))
  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")

  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      lanes_predicted = laneDetetctionAlgorithm(frame)
      lanes_predicted = cv2.cvtColor(lanes_predicted, cv2.COLOR_RGB2BGR)
      cv2.imshow('Lane Detection', lanes_predicted)
      output.write(lanes_predicted)
      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Break the loop
    else: 
      break

  # When everything done, release the video capture object
  cap.release()
  output.release()
  # Closes all the frames
  cv2.destroyAllWindows()