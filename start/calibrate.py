import os
import pickle

import cv2
import numpy as np


class CameraCalibrator:
    def __init__(self, calibration_images, no_corners_x_dir, no_corners_y_dir,
                 use_existing_camera_coefficients=True):
        """
        This class encapsulates camera calibration process. When creating an instance of
        CameraCalibrator class, if use_existing_camera_coefficients is False,  __calibrate()
        method is called and save camera calibration coefficients.
        :param calibration_images:
            The list of image used for camera calibration
        :param no_corners_x_dir:
            The number of horizontal corners in calibration images
        :param no_corners_y_dir:
            The number of vertical corners in calibration images
        """
        self.calibration_images = calibration_images
        self.no_corners_x_dir = no_corners_x_dir
        self.no_corners_y_dir = no_corners_y_dir
        self.object_points = []
        self.image_points = []

        if not use_existing_camera_coefficients:
            self._calibrate()

    def _calibrate(self):
        """
        :return:
            Camera calibration coefficients as a python dictionary
        """
        object_point = np.zeros((self.no_corners_x_dir * self.no_corners_y_dir, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:self.no_corners_x_dir, 0:self.no_corners_y_dir].T.reshape(-1, 2)

        for idx, file_name in enumerate(self.calibration_images):
            image = cv2.imread(file_name)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ret, corners = cv2.findChessboardCorners(gray_image,
                                                     (self.no_corners_x_dir, self.no_corners_y_dir),
                                                     None)
            if ret:
                self.object_points.append(object_point)
                self.image_points.append(corners)

        image_size = (image.shape[1], image.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points,
                                                           self.image_points, image_size, None, None)
        
        #print (ret)
        
        print(np.vectorize("%.2f".__mod__)(mtx))
        print(np.vectorize("%.2f".__mod__)(dist))
        
        #print (mtx)
        #print (dist)
        #print (rvecs)
        #print (tvecs)
        
        calibrated_data = {'mtx': mtx, 'dist': dist}

        #with open(CAMERA_CALIBRATION_COEFFICIENTS_FILE, 'wb') as f:
        #    pickle.dump(calibrated_data, file=f)

    def undistort(self, image):
        """
        :param image:
        :return:
        """

        if not os.path.exists(CAMERA_CALIBRATION_COEFFICIENTS_FILE):
            raise Exception('Camera calibration data file does not exist at ' +
                            CAMERA_CALIBRATION_COEFFICIENTS_FILE)

        with open(CAMERA_CALIBRATION_COEFFICIENTS_FILE, 'rb') as f:
            calibrated_data = pickle.load(file=f)

        # image = cv2.imread(image)
        return cv2.undistort(image, calibrated_data['mtx'], calibrated_data['dist'],
                             None, calibrated_data['mtx'])
    