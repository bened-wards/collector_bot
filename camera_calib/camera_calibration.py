# produced by Ben Edwards with reference to these materials:
# https://medium.com/@nflorent7/a-comprehensive-guide-to-camera-calibration-using-charuco-boards-and-opencv-for-perspective-9a0fa71ada5f
# https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import json
import os
import cv2
import numpy as np
from charuco_gen import * # just getting the constants

def get_calibration_parameters(img_dir):
    # Define the aruco dictionary, charuco board and detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    
    # Load images from directory
    image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]
    all_charuco_ids = []
    all_charuco_corners = []

    # Loop over images and extraction of corners
    for image_file in image_files:
        print("Processing: ", image_file)
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgSize = image.shape
        marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(image)
        
        if len(marker_ids) > 0: # If at least one marker is detected
            # cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)

            if charucoIds is not None and len(charucoCorners) > 3:
                all_charuco_corners.append(charucoCorners)
                all_charuco_ids.append(charucoIds)
    
    # Calibrate camera with extracted information
    result, camera_matrix, distortion_coefs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, imgSize, None, None)
    return camera_matrix, distortion_coefs

def get_optimal_camera_matrix_and_roi(camera_matrix, distortion_coefs, image_size):
    h, w = image_size
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefs, (w,h), 1, (w,h))
    return newcameramtx, roi

def undistort_image(image_path, camera_matrix, distortion_coefs):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape[:2]
    new_camera_matrix, roi = get_optimal_camera_matrix_and_roi(camera_matrix, distortion_coefs, (w,h))
    image = cv2.undistort(image, camera_matrix, distortion_coefs, None, new_camera_matrix)
    return image



if __name__ == "__main__":
    OUTPUT_JSON = 'calibration.json'

    camera_matrix, distortion_coefs = get_calibration_parameters(img_dir='./images/mounted_logitech_640_480_2/')
    data = {"camera_matrix": camera_matrix.tolist(), "distortion_coefs": distortion_coefs.tolist()}

    with open(OUTPUT_JSON, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f'Data has been saved to {OUTPUT_JSON}')

    # retrieve camera params
    json_file_path = './calibration.json'

    with open(json_file_path, 'r') as file: # Read the JSON file
        json_data = json.load(file)

    camera_matrix = np.array(json_data['camera_matrix'])
    distortion_coefs = np.array(json_data['distortion_coefs'])

    # get optimal camera matrix based on image dimensions
    # i.e. based on the free scaling parameter
    image_size = (640, 480)
    optimal_camera_matrix, roi = get_optimal_camera_matrix_and_roi(camera_matrix, distortion_coefs, image_size)
    print(optimal_camera_matrix)
    print(roi)
