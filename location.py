import cv2
import json
import numpy as np

# These coordinates are important for estimating the distance, they are of the same unit as the distance.
# These are virtual coordinates, the geometric reference originates from the center of the image.
RealityCoordinatePoints = np.array([
                      (-0.1, 0.1, 0),  # left top

                      (0.1, 0.1, 0),  # right top

                      (0.1, -0.1, 0),  # right bottom

                      (-0.1, -0.1, 0),  # left bottom
                     ], dtype="double")
class Location():
    def __init__(self, image_points, object_points=RealityCoordinatePoints, path_file_calibration="./images_to_calibration/calib_data_rational_init.json"):
        # Load calibration file
        self.data = json.load(open(path_file_calibration))

        # Reality keys
        self.object_points = object_points
        self.image_points = image_points

        self.camera_matrix = np.array(self.data['mtx'])
        self.distortion_coefficients = np.array(self.data["dist"])
        self.flags = cv2.SOLVEPNP_ITERATIVE

    def object_location(self):

        """To estimate location"""

        success, rotation_vector, translation_vector = cv2.solvePnP(objectPoints=self.object_points, imagePoints=self.image_points,
                                                        cameraMatrix=self.camera_matrix, distCoeffs=self.distortion_coefficients, flags=self.flags)
        if success:
            return translation_vector
        else:
            return None, None



