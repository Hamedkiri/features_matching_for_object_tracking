from tkinter import *
import cv2 as cv
import numpy as np
import random
import cv2
from location import Location


INDEX_AKAZE = 1
INDEX_BRISK = 2
INDEX_ORB = 3
INDEX_SIFT = 4
INDEX_SURF = 5
RectangleCalibrationTop_x = 500
RectangleCalibrationTop_y = 100
RectangleCalibrationBottom_x = 900
RectangleCalibrationBottom_y = 600



def nothing(x):
    pass


def select_algorithm(index):
    """To select different algorithm of features-matching."""
    if index == INDEX_AKAZE:
        # default threshold=0.00001, nOctaves=16
        return cv.AKAZE_create()
    elif index == INDEX_BRISK:
        return cv.BRISK_create()
    elif index == INDEX_ORB:
        # default nfeatures=10000, scaleFactor=1.2
        return cv.ORB_create(3000)
    elif index == INDEX_SIFT:
        return cv.SIFT_create()
    elif index == INDEX_SIFT:
        return cv2.SIFT_create()

def get_features(image, index):
    """Extraction of keypoints and descriptors in black-white image version."""
    image_to_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


    algorithm = select_algorithm(index=index)
    keypoints, descriptors = algorithm.detectAndCompute(image_to_gray, None)
    return keypoints, descriptors


def algorithms_of_matching_features(index, reference_descriptors, test_descriptors):
    """Matching between the descriptors"""
    if index == INDEX_AKAZE:
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_SL2)
    elif index == INDEX_BRISK  or index == INDEX_ORB:
        matcher = cv.BFMatcher(cv.NORM_HAMMING2)
    elif index == INDEX_SIFT:
        matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

    if index != INDEX_SIFT:
        keypoints_who_matches = matcher.knnMatch(reference_descriptors, test_descriptors, k=2)
    else:
        keypoints_who_matches = matcher.match(reference_descriptors, test_descriptors)

    return keypoints_who_matches

def search_good_match(index,keypoints_who_matches,ratio_test=0.8):
    """To select in the key points those corresponding to a lower distance ratio_test"""

    #print(keypoints_who_matches)
    best_matchs = []
    if index != INDEX_SIFT:
        for m, n in keypoints_who_matches:
            if (m.distance < ratio_test * n.distance):
                best_matchs.append(m)
    else:
        for m in keypoints_who_matches:
            for n in keypoints_who_matches:
                if (m.distance < ratio_test * n.distance):
                    best_matchs.append(m)

    return best_matchs

def search_homography_between_images(reference_image,test_image,best_matchs,reference_keypoints,test_keypoints):
    """ To search homography between the two images """
    new_reference_image = reference_image
    reference_image_to_gray = cv.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    test_image_to_gray = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
    shape_reference_image = reference_image_to_gray.shape
    yes = False
    try:
        if len(best_matchs) >= 4:
            src_pts = np.float32([reference_keypoints[m.queryIdx].pt for m in best_matchs]).reshape(-1, 1, 2)
            dst_pts = np.float32([test_keypoints[m.trainIdx].pt for m in best_matchs]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            h, w = reference_image_to_gray.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            if M is not None:
                dst = cv.perspectiveTransform(pts, M)
                #print(np.int32(dst))
                if len(np.int32(dst)) == 4:
                    x1 = np.int32(dst)[0][0][0]
                    y1 = np.int32(dst)[0][0][1]  # left top
                    x2 = np.int32(dst)[1][0][0]
                    y2 = np.int32(dst)[1][0][1]  # right top
                    x3 = np.int32(dst)[2][0][0]
                    y3 = np.int32(dst)[2][0][1]  # right bottom
                    x4 = np.int32(dst)[3][0][0]
                    y4 = np.int32(dst)[3][0][1]  # left bottom
                    #print(np.int32(dst))

                    image_points_2D = np.array([
                        (x1, y1),  # left top

                        (x2, y2),  # right top

                        (x3, y3),  # right bottom

                        (x4, y4),  # left bottom

                    ], dtype="double")

                    geolocation = Location(image_points=image_points_2D)

                oui = []
                for a in best_matchs:
                    oui.append(test_keypoints[a.trainIdx])
                test_image_with_draw_keypoints = cv.drawKeypoints(test_image, oui, None,
                                                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                image_with_homography = cv.polylines(test_image_with_draw_keypoints, [np.int32(dst)], True,
                                                     (255, 0, 0), 3)

                if len(np.int32(dst)) == 4:


                    max_width_rectangle = max(x4 - x1, x3 - x1)
                    max_length_rectangle = max(y2-y1, y3-y4)
                    min_width_rectangle = min(x4 - x1, x3 - x1)
                    min_length_rectangle = min(y2-y1, y3-y4)


                    #print(length_rectangle)
                    #print(width_rectangle)
                    #print(shape_reference_image)
                    # and (max_width_rectangle > int(shape_reference_image[0]/2)) and (max_length_rectangle > int(shape_reference_image[1]/2)) :

                    if (max_length_rectangle != 0) and (max_width_rectangle != 0) and (0.8 < min_width_rectangle/max_width_rectangle) \
                        and (0.8 < min_length_rectangle/max_length_rectangle) and (max_width_rectangle > int(shape_reference_image[0]/3)) \
                        and (max_length_rectangle > int(shape_reference_image[1]/3)):

                        yes = True
                        #print((x4 - x1, x3 - x1))
                        #print((y2 - y1, y3 - y4))
                        #print(shape_reference_image)
                        #print("(x1,y1)=" + str((x1, y1)), "(x2,y2)=" + str((x2, y2)), "(x3,y3)=" + str((x3, y3)),
                             # "(x4,y4)=" + str((x4, y4)))

                        new_reference_image = test_image[y1:y3, x1:x3]
                        #location = Location(image_points=[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).object_location()
                        #print("test:"+str(location))

                        #print(test_image_to_gray[x1:x2, y2:y1])
                        image_with_homography = cv2.rectangle(test_image_with_draw_keypoints, (x1, y1), (x3, y3),
                                                             (0, 255, 0), 3)
                        #image_with_homography = geolocation.draw_line(image=image_with_homography)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        distance_of_camera = round(np.linalg.norm(geolocation.object_location(), ord=2),2)
                        image_with_homography = cv2.putText(image_with_homography, str(distance_of_camera)+"m", (50, 50), font, 1,
                                                            (0, 255, 255), 2, cv2.LINE_4)

                        #print(geolocation.object_location())
                    else:
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


"""def random_choice_submatrix(image, width, height):

    MinimalSizeScreen = 250
    SizeSubMatrix = 200
    min_mid = min(int(width/2), int(height/2))
    # print(min_mid)
    if width > MinimalSizeScreen and height > MinimalSizeScreen:
        i = random.randint(0, min_mid)

        return image[500: 900, 100: 600]
    else:
        print("matrice de trop petite taille")
        return None"""


def chessBoartDetector(image):

    """Algorithm to recognize chessboart"""
    #NumberOfSubMatrix = 1

    WhiteBlackRatio = 0.5  # Ratio to consider that pixel is white or black

    Interval = 0.1

    count_white = 0
    count_black = 0
    probability = [None]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    image = image / image.max()
    image = image[RectangleCalibrationTop_x: RectangleCalibrationBottom_x,  RectangleCalibrationTop_y: RectangleCalibrationBottom_y]
    width = image.shape[0]
    height = image.shape[1]





    for i in range(width):
        for j in range(height):
            if image[i, j] > WhiteBlackRatio:
                count_white = count_white + 1
            else:
                count_black = count_black + 1

    probability[0] = min(count_white, count_black)/max(count_white, count_black)
    print(probability)
    if probability[0] > 0.99:
        return 1
    else:
        return 0







