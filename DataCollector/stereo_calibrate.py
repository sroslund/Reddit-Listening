import glob
import cv2
import numpy as np
import pathlib
import os

def calibrate_camera(dir_path, image_format, square_size, width, height):
    '''Calibrate a camera using chessboard images.'''
    # termination criteria
    gray = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    #images = pathlib.Path(dir_path).glob(f'*.{image_format}')
    files = os.listdir(dir_path)

    # filter files to only include those ending in .jpg
    images = [file for file in files if file.endswith(".jpg")]

    # Iterate through all images
    for fname in images:
        print("here")
        img = cv2.imread(dir_path+'/'+str(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)
 
    return mtx, dist
 
mtx_l, dist_l = calibrate_camera('./left','.jpg',3,5,7)
mtx_r, dist_r = calibrate_camera('./right','.jpg',3,5,7)

def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    #read the synched frames
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)
    c1_images_names = images_names[:len(images_names)//2]
    c2_images_names = images_names[len(images_names)//2:]
 
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv2.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv2.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    rows = 5 #number of checkerboard rows.
    columns = 7 #number of checkerboard columns.
    world_scaling = 3. #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        print(frame1, frame2)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (5, 7), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (5, 7), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            cv2.drawChessboardCorners(frame1, (5,7), corners1, c_ret1)
            #cv2.imshow('img', frame1)
            #k = cv2.waitKey(1000)
            cv2.drawChessboardCorners(frame2, (5,7), corners2, c_ret2)
            #cv2.imshow('img2', frame2)
            #k = cv2.waitKey(1000)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
 
    print("almost done")
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 

R, T = stereo_calibrate(mtx_l, dist_l, mtx_r, dist_r, 'joined/*.jpg')

print(R, T)

img = cv2.imread('input2.jpg')

# Get image dimensions
height, width = img.shape[:2]
img_l = img[:, :int(width / 2)]
img_r = img[:, int(width / 2):]

print(img_l.shape)

height, width = img_l.shape[:2]
# Scale down the image by half
#img_l = cv2.resize(img_l, (int(width/4), int(height/4)), interpolation=cv2.INTER_AREA)
#img_r = cv2.resize(img_r, (int(width/4), int(height/4)), interpolation=cv2.INTER_AREA)
cv2.imshow('Depth Map', img_l)
#cv2.imshow('Depth Map', img_r)
cv2.waitKey(0)
cv2.imshow('Depth Map', img_r)
cv2.waitKey(0)
cv2.destroyAllWindows()

def compute_left_disparity_map(img_left, img_right):
    
    ### START CODE HERE ###
    img_left_g = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_g = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    matcher = cv2.StereoBM_create(6*16, 31)
    
    out = matcher.compute(img_left_g, img_right_g)
    
    out = out.astype(np.float32)/16
    
    ### END CODE HERE ###
    
    return out

#print(compute_left_disparity_map(img_l, img_r))

#cv2.imshow('Depth Map',compute_left_disparity_map(img_l, img_r))
#cv2.waitKey(0)


# Rectify the stereo images
rectification_flags = cv2.CALIB_ZERO_DISPARITY
# Compute the rectification maps

print(img_l.shape)
print(img_r.shape)

rectification_map_left, rectification_map_right, projection_matrix_left, projection_matrix_right, Q, roi_left, roi_right = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, img_l.shape, R, T, rectification_flags, alpha=1)
print(rectification_map_left)
#rectification_map_left = rectification_map_left.astype(np.float32)
#rectification_map_right = rectification_map_right.astype(np.float32)

# Rectify the left and right images using the rectification maps
#left_img_rectified = cv2.remap(img_l, rectification_map_left, rectification_map_right, cv2.INTER_LINEAR)
#right_img_rectified = cv2.remap(img_r, rectification_map_left, rectification_map_right, cv2.INTER_LINEAR)
#print(left_img_rectified)
l_maps = cv2.initUndistortRectifyMap(mtx_l, dist_l, rectification_map_left, projection_matrix_left, img_l.shape, cv2.CV_16SC2)
r_maps = cv2.initUndistortRectifyMap(mtx_r, dist_r, rectification_map_right, projection_matrix_right, img_l.shape, cv2.CV_16SC2)
left_img_rectified = cv2.remap(img_l, l_maps[0], l_maps[1], cv2.INTER_LINEAR)
right_img_rectified = cv2.remap(img_r, r_maps[0], r_maps[1], cv2.INTER_LINEAR)
left_img_rectified = cv2.convertScaleAbs(cv2.cvtColor(left_img_rectified, cv2.COLOR_BGR2GRAY))
right_img_rectified = cv2.convertScaleAbs(cv2.cvtColor(right_img_rectified, cv2.COLOR_BGR2GRAY))
print(left_img_rectified)

cv2.imshow('Depth Map', left_img_rectified)
#cv2.imshow('Depth Map', img_r)
cv2.waitKey(0)
cv2.imshow('Depth Map', right_img_rectified)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Compute the disparity map
stereo = cv2.StereoBM_create(numDisparities=20*16, blockSize=23)
disparity = stereo.compute(left_img_rectified, right_img_rectified)

# Convert the disparity map to a depth map
focal_length = mtx_l[0][0]
baseline = abs(T[0])
depth_map = np.zeros_like(disparity, dtype=np.float32)
for i in range(depth_map.shape[0]):
    for j in range(depth_map.shape[1]):
        if disparity[i][j] > 0:
            depth_map[i][j] = focal_length * baseline / disparity[i][j]

# Display the depth map
cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()






