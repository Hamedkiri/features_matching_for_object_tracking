import cv2
import json
import numpy as np

RealityCoordinatePoints = points_3D = np.array([
                      (-0.1, 0.1, 0),  # left top

                      (0.1, 0.1, 0),  # right top

                      (0.1, -0.1, 0),  # right bottom

                      (-0.1, -0.1, 0),  # left bottom
                     ], dtype="double")
class Location():
    def __init__(self, image_points, object_points=RealityCoordinatePoints, path_file_calibration="./images_to_calibration/calib_data_rational_init.json"):

        self.data = json.load(open(path_file_calibration))
        self.object_points = object_points
        self.image_points = image_points
        self.camera_matrix = np.array(self.data['mtx'])
        self.distortion_coefficients = np.array(self.data["dist"])
        self.flags = cv2.SOLVEPNP_ITERATIVE

    def object_location(self):
        success, rotation_vector, translation_vector = cv2.solvePnP(objectPoints=self.object_points, imagePoints=self.image_points,
                                                        cameraMatrix=self.camera_matrix, distCoeffs=self.distortion_coefficients, flags=self.flags)
        if success:
            return translation_vector
        else:
            return None, None

    def draw_image_points_2D(self, image):
        for point in self.image_points:
            cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

    def draw_line(self, image):
        rotation_vector, translation_vector = self.object_location()

        if rotation_vector is not None:
            point2D_results, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                       translation_vector, self.camera_matrix, self.distortion_coefficients)

        for point_before in self.image_points:
            for point_after in point2D_results:
                image = cv2.line(image, point_before, point_after, (255, 255, 255), 2)
        return image



# print(CoordinatePoints.shape)
# print(points_img.shape)
# local = Location(image_points=points_img).object_location()
# print(local)