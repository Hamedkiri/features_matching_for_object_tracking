import numpy as np
import cv2
import glob
import os
import json

i = 0


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Calibration_of_camera():

    def __init__(self, path_images_calibration="./images_to_calibration"):
        self.path_images_calibration = path_images_calibration
        self.grid_size_width = 4
        self.grid_size_height = 4
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.grid_size_width  * self.grid_size_height, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.grid_size_width, 0:self.grid_size_height].T.reshape(-1, 2)  # multiply by square size in mm ? 45*

        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.image_size = tuple()

        self.images = glob.glob(os.path.join(self.path_images_calibration, '*.jpg'))
        self.image_to_gray = None

    def show_coners(self):
        for fname in self.images:
            #print(fname)
            img = cv2.imread(fname)
            #print(img)
            self.image_size = img.shape

            self.image_to_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(self.image_to_gray, (self.grid_size_width, self.grid_size_height), None)
            # If found, add object points, image points (after refining them)
            if ret:
                print("corners detected")
                self.objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(self.image_to_gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners)
                # Draw and display the corners

                cv2.drawChessboardCorners(img, (self.grid_size_width, self.grid_size_height), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(10)
            else:
                print("corners not detected")

        cv2.destroyAllWindows()

    def make_calibration(self):
        init_mtx = np.array([
            [
                1071,
                0.0,
                960
            ],
            [
                0.0,
                1071,
                540
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ])

        print("Calibrating camera with radial / tangential model ...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.image_to_gray.shape[::-1], init_mtx, None,
                                                           flags=cv2.CALIB_USE_INTRINSIC_GUESS)  # np.zeros(5), flags=cv.CALIB_FIX_K3 + cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K5) # flags=cv.CALIB_RATIONAL_MODEL)
        # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("Radial_tangential_calibration: ")
        print("ret: ", ret)
        print("mtx: ", mtx)
        print("dist: ", dist)

        print("Saving calib data to calib_data.json")
        with open(os.path.join(self.path_images_calibration, "calib_data_init.json"), "w+") as f:
            json.dump({"image_size": self.image_size, "reprojection error": ret, "mtx": mtx.tolist(), "dist": dist.tolist()},
                      f,
                      indent=4, cls=NumpyEncoder)

        print("Calibrating camera with rational model ...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.image_to_gray.shape[::-1], init_mtx,
                                                           None,
                                                           flags=cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_USE_INTRINSIC_GUESS)
        print("Rational_calibration: ")
        print("ret: ", ret)
        print("mtx: ", mtx)
        print("dist: ", dist)

        print("Saving calib data to calib_data_rational_init.json")
        with open(os.path.join(self.path_images_calibration, "calib_data_rational_init.json"), "w+") as f:
            json.dump({"image_size": self.image_size, "reprojection error": ret, "mtx": mtx.tolist(), "dist": dist.tolist()},
                      f,
                      indent=4, cls=NumpyEncoder)



Ty = Calibration_of_camera()
Ty.show_coners()
Ty.make_calibration()