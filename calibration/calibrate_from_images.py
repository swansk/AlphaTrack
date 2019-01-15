import numpy as np
import cv2
import glob
import json

# Dimensions on checkerboard
col_checker = 9
row_checker = 6

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((row_checker*col_checker, 3), np.float32)
objp[:, :2] = np.mgrid[0:col_checker,0:row_checker].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('../calibration_images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (6,9), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (6,9), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(50)

ret, camera_matrix, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

data = {"camera_matrix": camera_matrix, "dist_coeff": dist_coeff, "rvecs": rvecs, "tvecs": tvecs}

print(data)

np.save("calibration_files/camera_matrix.npy", camera_matrix)
np.save("calibration_files/dist_coeff.npy", dist_coeff)
np.save("calibration_files/rvecs.npy", rvecs)
np.save("calibration_files/tvecs.npy", tvecs)

cv2.destroyAllWindows()