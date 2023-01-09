import cv2 as cv
import numpy as np
import cv2
from location import Location

# INDEX which to inventory the different algorithms
INDEX = {"AKAZE": 1, "BRISK": 2, "ORB": 3, "SIFT": 4, "SURF": 5}


def select_algorithm(index):

    """To select different algorithm of features-matching."""

    if index == INDEX["AKAZE"]:
        return cv.AKAZE_create()
    elif index == INDEX["BRISK"]:
        return cv.BRISK_create()
    elif index == INDEX["ORB"]:
        return cv.ORB_create(3000)
    elif index == INDEX["SIFT"]:
        return cv.SIFT_create()


def get_features(image, index):

    """Extraction of keypoints and descriptors in black-white image version."""

    image_to_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    algorithm = select_algorithm(index=index)

    #  Extracting features and descriptions in the image
    keypoints, descriptors = algorithm.detectAndCompute(image_to_gray, None)

    return keypoints, descriptors


def algorithms_of_matching_features(index, reference_descriptors, test_descriptors):

    """Matching between the descriptors with different methods"""

    if index == INDEX["AKAZE"]:
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_SL2)

    elif index == INDEX["ORB"] or index == INDEX["BRISK"]:
        #print("orb")
        matcher = cv.BFMatcher(cv.NORM_HAMMING2)

    elif index == INDEX["SIFT"]:
        matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    if index != INDEX["SIFT"]:
        keypoints_who_matches = matcher.knnMatch(reference_descriptors, test_descriptors, k=2)
    else:
        keypoints_who_matches = matcher.match(reference_descriptors, test_descriptors)

    return keypoints_who_matches


def search_good_match(index, keypoints_who_matches, ratio_test=0.8):

    """Select keypoints whose distance(difference) is less than the threshold(ratio_test)"""

    best_matchs = []
    if index != INDEX["SIFT"]:
        for m, n in keypoints_who_matches:
            if m.distance < ratio_test * n.distance:
                best_matchs.append(m)
    else:
        for m in keypoints_who_matches:
            for n in keypoints_who_matches:
                if m.distance < ratio_test * n.distance:
                    best_matchs.append(m)

    return best_matchs


def search_homography_between_images(reference_image, test_image, best_matchs, reference_keypoints, test_keypoints):

    """ To search homography between the two images """

    new_reference_image = reference_image
    reference_image_to_gray = cv.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    # test_image_to_gray = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
    shape_reference_image = reference_image_to_gray.shape
    yes = False
    try:
        # To check if the number of keypoint exceeds four
        if len(best_matchs) >= 4:
            src_pts = np.float32([reference_keypoints[m.queryIdx].pt for m in best_matchs]).reshape(-1, 1, 2)
            dst_pts = np.float32([test_keypoints[m.trainIdx].pt for m in best_matchs]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            h, w = reference_image_to_gray.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            if M is not None:
                dst = cv.perspectiveTransform(pts, M)
                # print(np.int32(dst))
                if len(np.int32(dst)) == 4:
                    # left top
                    x1 = np.int32(dst)[0][0][0]
                    y1 = np.int32(dst)[0][0][1]
                    # right top
                    x2 = np.int32(dst)[1][0][0]
                    y2 = np.int32(dst)[1][0][1]
                    # right bottom
                    x3 = np.int32(dst)[2][0][0]
                    y3 = np.int32(dst)[2][0][1]
                    # left bottom
                    x4 = np.int32(dst)[3][0][0]
                    y4 = np.int32(dst)[3][0][1]

                    image_points_2D = np.array([
                        (x1, y1),  # left top

                        (x2, y2),  # right top

                        (x3, y3),  # right bottom

                        (x4, y4),  # left bottom

                    ], dtype="double")

                    # Distance estimation
                    geolocation = Location(image_points=image_points_2D)

                oui = []
                for a in best_matchs:
                    oui.append(test_keypoints[a.trainIdx])
                test_image_with_draw_keypoints = cv.drawKeypoints(test_image, oui, None,
                                                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # To draw homography in red color
                image_with_homography = cv.polylines(test_image_with_draw_keypoints, [np.int32(dst)], True,
                                                     (255, 0, 0), 3)

                if len(np.int32(dst)) == 4:
                    # Here, I check if the homography is a rectangle
                    max_width_rectangle = max(x4 - x1, x3 - x1)
                    max_length_rectangle = max(y2 - y1, y3 - y4)
                    min_width_rectangle = min(x4 - x1, x3 - x1)
                    min_length_rectangle = min(y2 - y1, y3 - y4)

                    # Criteria to check if homography is rectangle
                    if (max_length_rectangle != 0) and (max_width_rectangle != 0) and (
                            0.8 < min_width_rectangle / max_width_rectangle) \
                            and (0.8 < min_length_rectangle / max_length_rectangle) and (
                            max_width_rectangle > int(shape_reference_image[0] / 3)) \
                            and (max_length_rectangle > int(shape_reference_image[1] / 3)):

                        # To check if homography is a rectangle
                        yes = True

                        new_reference_image = test_image[y1:y3, x1:x3]
                        # To draw homography in green color
                        image_with_homography = cv.polylines(test_image_with_draw_keypoints, [np.int32(dst)], True,
                                                             (0, 255, 0), 3)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        distance_of_camera = round(np.linalg.norm(geolocation.object_location(), ord=2), 2)
                        image_with_homography = cv2.putText(image_with_homography, str(distance_of_camera) + "m",
                                                            (50, 50), font, 1,
                                                            (0, 255, 255), 2, cv2.LINE_4)

                    else:
                        # To check if homography is a rectangle
                        yes = False

                else:

                    new_reference_image = reference_image

                return new_reference_image, image_with_homography, yes
            else:
                return new_reference_image, test_image, yes
        else:
            return new_reference_image, test_image, yes
    except:
        return new_reference_image, test_image, yes
