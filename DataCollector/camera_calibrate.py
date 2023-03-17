from matplotlib import pyplot as plt
import pdb
import numpy as np
import cv2
import glob
import sys

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
    return cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 



# These 2 lists are for later usage after chessboard found.
l_goodpair = []
r_goodpair = []

# The termination criteria and flags are checked out from opencv sample (in calling order).
chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
cornersub_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 0.01)
stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
stereocalib_flags = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_SAME_FOCAL_LENGTH + \
        cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5

def minDispsCallBack(x):
    pass
def numDispsCallBack(x):
    pass
def bSizeCallBack(x):
    pass
def wSizeCallBack(x):
    pass
def disp12CallBack(x):
    pass
def uniqCallBack(x):
    pass
def spWCallBack(x):
    pass
def spRCallBack(x):
    pass

def tuneDisparity(lframe, rframe, l_maps, r_maps, focal_length):
        
    # use the rectified data to do remap on webcams
    lframe_remap = cv2.remap(lframe, l_maps[0], l_maps[1], cv2.INTER_LINEAR)
    rframe_remap = cv2.remap(rframe, r_maps[0], r_maps[1], cv2.INTER_LINEAR)

    minDisp = cv2.getTrackbarPos('minDisparity', 'disparity_parameters')
    numDisp = cv2.getTrackbarPos('numDisparities', 'disparity_parameters') * 16
    blockSize = cv2.getTrackbarPos('blockSize', 'disparity_parameters')
    blockSize = blockSize > 1 and blockSize or 2
    SADWindowSize = cv2.getTrackbarPos('SADWindowSize', 'disparity_parameters')
    P1 = 8*3*SADWindowSize**2
    P2 = 32*3*SADWindowSize**2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disparity_parameters')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disparity_parameters')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disparity_parameters')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disparity_parameters')

    stereo = cv2.StereoSGBM_create(\
            minDisparity=minDisp,
            numDisparities=numDisp,
            blockSize=blockSize,
            P1=P1,
            P2=P2,
            disp12MaxDiff=disp12MaxDiff,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange)

    disparity = stereo.compute(lframe_remap, rframe_remap).astype(np.float32) / 16.0
    optimal_disparity = (disparity - minDisp) / numDisp
    cv2.imshow('disparaty_map', optimal_disparity)

    baseline_cam = 9.3
    distance = baseline_cam * focal_length / optimal_disparity

    print ("distance in center: ", distance[320][240]/100)
    
    cv2.imshow('left_webcam remap', lframe_remap)
    cv2.imshow('right_webcam remap', rframe_remap)


def stereoRectificationProcess(rectify_scale, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F):

    # call cv2.stereoRectify to get the rectification parameters
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify( \
            cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (640, 480), R, T, alpha=rectify_scale)
    # prepare to remap the webcams
    l_maps = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (640, 480), cv2.CV_16SC2)
    r_maps = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (640, 480), cv2.CV_16SC2)
    #with open('lmaps.txt', 'wb') as f:
    #    np.savetxt(f, l_maps[0])
    #    np.savetxt(f, l_maps[1])

    focal_length_x = cameraMatrix1[0][0]
    #focal_length_y = cameraMatrix1[1][1]

    """
    for l, r in zip(l_goodpair, r_goodpair):
        #l_gray = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
        #r_gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        l_imgremap = cv2.remap(l, l_maps[0], l_maps[1], cv2.INTER_LINEAR)
        r_imgremap = cv2.remap(r, r_maps[0], r_maps[1], cv2.INTER_LINEAR)
        cv2.imshow('left image_remap', l_imgremap)
        cv2.imshow('right image_remap', r_imgremap)
        cv2.waitKey(5000)
    cv2.destroyAllWindows()
    """

    # read in webcams' frames and prepare for disparity computing parameter tuning
    lcap = cv2.VideoCapture(1)
    cv2.namedWindow('disparity_parameters')
    text = np.zeros((5, 500), dtype=np.uint8)
    cv2.imshow('disparity_parameters', text)
    cv2.createTrackbar('minDisparity', 'disparity_parameters', 16, 100, minDispsCallBack)
    cv2.createTrackbar('numDisparities', 'disparity_parameters', 1, 20, numDispsCallBack)    # divisible by 16
    cv2.createTrackbar('blockSize', 'disparity_parameters', 7, 30, bSizeCallBack)            # odd number, 1 < 3 < blockSize < 11
    cv2.createTrackbar('SADWindowSize', 'disparity_parameters', 3, 30, wSizeCallBack)        
    #cv2.createTrackbar('P1', 'disparity_parameters', 1, 1, p1SizeCallBack)
    #cv2.createTrackbar('P2', 'disparity_parameters', 1, 1, p2SizeCallBack)
    cv2.createTrackbar('disp12MaxDiff', 'disparity_parameters', 1, 30, disp12CallBack)
    cv2.createTrackbar('uniquenessRatio', 'disparity_parameters', 1, 30, uniqCallBack)
    cv2.createTrackbar('speckleWindowSize', 'disparity_parameters', 100, 200, spWCallBack)   # 55 < speckleWindow < 200
    cv2.createTrackbar('speckleRange', 'disparity_parameters', 1, 32, spRCallBack)           # 1 <= speckleRange <= 2

    while(True):

        lret, frame = lcap.read()
        height, width = frame.shape[:2]
        lframe = frame[:, :int(width / 2)]
        rframe = frame[:, int(width / 2):]
        
        # show disparity map and tune the paramters in real-time
        tuneDisparity(lframe, rframe, l_maps, r_maps, focal_length_x)

        key = cv2.waitKey(5)&0xFF
        if key == 27 or key == ord('q'):
            print('bye')
            break

    lcap.release()
    cv2.destroyAllWindows()
    # done
    
def getCalibratefromStereoImage(objpoints, l_imgpoints, r_imgpoints):
    
    # from OpenCV docs: if any of CV_CALIB_FIX_ASPECT_RATIO... are specified, the matrix components must be initialized.
    cameraMatrix1 = cv2.initCameraMatrix2D(objpoints, l_imgpoints, (640, 480), 0)
    cameraMatrix2 = cv2.initCameraMatrix2D(objpoints, r_imgpoints, (640, 480), 0)
    distCoeffs1 = None
    distCoeffs2 = None

    # directly call stereCalibrate from OpenCV library
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
            cv2.stereoCalibrate(objpoints, l_imgpoints, r_imgpoints, \
            cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (640, 480), \
            criteria=stereocalib_criteria, flags=stereocalib_flags)

    return retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F

def drawChessboard(height, width):
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) for 3-d use,
    # which means we don't really need
    objp = np.zeros((height*width,3), np.float32)
    objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space

    l_imgpoints = [] # 2d points in left image plane.
    r_imgpoints = [] # 2d points in right image plane.
    
    # count how many pairs for for loop to loop through
    l_images = glob.glob('images/left*.png')

    for cnt in range(1, len(l_images)+1):
        l_img = cv2.imread('images/left'+str(cnt)+'.png')
        r_img = cv2.imread('images/right'+str(cnt)+'.png')

        l_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
        r_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

        # First: find chessboard corners by cv2.findChessboardCorners.
        l_ret, l_corners = cv2.findChessboardCorners(l_gray, (width, height), chessboard_flags)
        r_ret, r_corners = cv2.findChessboardCorners(r_gray, (width, height), chessboard_flags)

        if l_ret and r_ret:
            print ("Found image pair:", cnt)
            l_goodpair.append(l_img)
            r_goodpair.append(r_img)
            
            # Second: find subpixel coordinates by cv2.cornerSubPix.
            cv2.cornerSubPix(l_gray, l_corners, (11, 11), (-1, -1), cornersub_criteria)
            cv2.cornerSubPix(r_gray, r_corners, (11, 11), (-1, -1), cornersub_criteria)
            
            # Third: store keypoints.
            l_imgpoints.append(l_corners)
            r_imgpoints.append(r_corners)
            objpoints.append(objp)

    cv2.destroyAllWindows()
    return l_imgpoints, r_imgpoints, objpoints

if __name__ == '__main__':



    print ('usage: python calib.py height width (default 7, 10)')
    
    # Step 1: For each stereo pair we need to find the chessboard and store the keypoints.
    #l_imgpoints, r_imgpoints, objpoints = drawChessboard(7, 10)

    print ('calibrating...')
    # Step 2: Compute calibration.
    mtx_l, dist_l = calibrate_camera('./left','.jpg',3,5,7)
    mtx_r, dist_r = calibrate_camera('./right','.jpg',3,5,7)
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = stereo_calibrate(mtx_l, dist_l, mtx_r, dist_r, 'joined/*.jpg')
    
    #pdb.set_trace()
    # Step 2.5: Save the calibration stats to disk for future use
    """
    with open('c1d1c2d2RTEF.txt', 'wb') as f:
        #f.write("cameraMatrix1:\n")
        np.savetxt(f, cameraMatrix1)
        #f.write("distCoeffs1:\n")
        np.savetxt(f, distCoeffs1)
        #f.write("cameraMatrix2:\n")
        np.savetxt(f, cameraMatrix2)
        #f.write("distCoeffs2:\n")
        np.savetxt(f, distCoeffs2)
        #f.write("R:\n")
        np.savetxt(f, R)
        #f.write("T:\n")
        np.savetxt(f, T)
        #f.write("E:\n")
        np.savetxt(f, E)
        #f.write("F:\n")
        np.savetxt(f, F)
    """

    print ('rectifying...')
    # Step 3: Stereo rectification
    if len(sys.argv) == 2:
        rectify_scale = float(sys.argv[1])
    else:
        rectify_scale = 0
    stereoRectificationProcess(rectify_scale, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F)